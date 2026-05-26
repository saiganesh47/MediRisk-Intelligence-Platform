"""
MediRisk Intelligence Platform
Fraud Detection Model — Training & Evaluation
Uses PySpark MLlib: Random Forest Classifier with cross-validation,
class-imbalance handling via stratified sampling, and full eval metrics.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from pyspark.ml import Pipeline
from pyspark.ml.classification import (
    GBTClassifier,
    RandomForestClassifier,
)
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.feature import (
    Imputer,
    MinMaxScaler,
    StringIndexer,
    VectorAssembler,
)
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ─── Feature Columns ──────────────────────────────────────────────────────────

NUMERIC_FEATURES = [
    "billed_amount",
    "allowed_amount",
    "paid_amount",
    "lag_days",
    "billed_to_allowed_ratio",
    "billed_vs_provider_avg",
    "prov_90d_claim_count",
    "prov_90d_billed_total",
    "prov_90d_avg_billed",
    "prov_90d_stddev_billed",
    "prov_90d_unique_patients",
    "prov_90d_fraud_count",
    "prov_total_claims",
    "prov_fraud_rate",
    "prov_avg_lag_days",
    "pat_30d_claim_count",
    "pat_30d_billed_total",
    "pat_30d_unique_providers",
    "pat_total_claims",
    "pat_avg_billed",
    "fraud_signals_count",
    "is_high_lag",
    "is_weekend_service",
]

CATEGORICAL_FEATURES = [
    "diagnosis_code",
    "procedure_code",
    "claim_status",
]

LABEL_COL    = "label"
FEATURES_COL = "features"


# ─── Session ──────────────────────────────────────────────────────────────────

def get_spark(app_name: str = "MediRisk-FraudModel") -> SparkSession:
    return (
        SparkSession.builder
        .appName(app_name)
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate()
    )


# ─── Data Preparation ─────────────────────────────────────────────────────────

def load_and_prepare(spark: SparkSession, gold_path: str) -> tuple[DataFrame, DataFrame]:
    """
    Load Gold claims features, cast label, handle class imbalance
    via stratified under-sampling, and split train/test.
    """
    df = (
        spark.read
        .format("delta")
        .load(f"{gold_path}/claims_features")
        .select(*NUMERIC_FEATURES, *CATEGORICAL_FEATURES, "is_fraud")
        .withColumn(LABEL_COL, F.col("is_fraud").cast("double"))
        .drop("is_fraud")
        .dropDuplicates()
    )

    total    = df.count()
    pos      = df.filter(F.col(LABEL_COL) == 1).count()
    neg      = df.filter(F.col(LABEL_COL) == 0).count()
    pos_rate = pos / total
    logger.info(
        "Dataset: %d total  |  %d fraud (%.2f%%)  |  %d non-fraud",
        total, pos, pos_rate * 100, neg,
    )

    # Under-sample majority class to achieve ~20% fraud rate for training
    target_ratio    = 0.20
    desired_neg     = int(pos / target_ratio * (1 - target_ratio))
    neg_fraction    = min(1.0, desired_neg / neg)

    fraud_df        = df.filter(F.col(LABEL_COL) == 1)
    non_fraud_df    = df.filter(F.col(LABEL_COL) == 0).sample(neg_fraction, seed=42)
    balanced_df     = fraud_df.union(non_fraud_df)

    logger.info(
        "Balanced dataset: %d rows  (fraud=%d, non-fraud=%d)",
        balanced_df.count(), fraud_df.count(), non_fraud_df.count(),
    )

    # Stratified-ish split — split each class separately then union
    train_fraud, test_fraud       = fraud_df.randomSplit([0.8, 0.2], seed=42)
    train_non, test_non           = non_fraud_df.randomSplit([0.8, 0.2], seed=42)

    train_df = train_fraud.union(train_non)
    test_df  = test_fraud.union(test_non)

    logger.info("Train: %d  |  Test: %d", train_df.count(), test_df.count())
    return train_df, test_df


# ─── Pipeline Construction ────────────────────────────────────────────────────

def build_pipeline(model_type: str = "rf") -> Pipeline:
    """
    Assembles a full ML pipeline:
    StringIndexer → Imputer → VectorAssembler → MinMaxScaler → Classifier
    """

    # Encode categoricals
    indexers = [
        StringIndexer(
            inputCol=col,
            outputCol=f"{col}_idx",
            handleInvalid="keep",
        )
        for col in CATEGORICAL_FEATURES
    ]

    # Impute missing numerics with median
    imputer = Imputer(
        inputCols=NUMERIC_FEATURES,
        outputCols=[f"{c}_imp" for c in NUMERIC_FEATURES],
        strategy="median",
    )
    imputed_cols = [f"{c}_imp"    for c in NUMERIC_FEATURES]
    indexed_cols = [f"{c}_idx"    for c in CATEGORICAL_FEATURES]

    assembler = VectorAssembler(
        inputCols=imputed_cols + indexed_cols,
        outputCol="raw_features",
        handleInvalid="keep",
    )

    scaler = MinMaxScaler(inputCol="raw_features", outputCol=FEATURES_COL)

    if model_type == "rf":
        classifier = RandomForestClassifier(
            labelCol=LABEL_COL,
            featuresCol=FEATURES_COL,
            predictionCol="prediction",
            probabilityCol="probability",
            rawPredictionCol="rawPrediction",
            numTrees=200,
            maxDepth=10,
            featureSubsetStrategy="sqrt",
            seed=42,
        )
    elif model_type == "gbt":
        classifier = GBTClassifier(
            labelCol=LABEL_COL,
            featuresCol=FEATURES_COL,
            predictionCol="prediction",
            maxIter=100,
            maxDepth=6,
            stepSize=0.1,
            seed=42,
        )
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    return Pipeline(stages=indexers + [imputer, assembler, scaler, classifier])


# ─── Cross-Validation ─────────────────────────────────────────────────────────

def train_with_cv(
    pipeline:    Pipeline,
    train_df:    DataFrame,
    model_type:  str = "rf",
    num_folds:   int = 3,
) -> object:
    evaluator = BinaryClassificationEvaluator(
        labelCol=LABEL_COL,
        metricName="areaUnderROC",
    )

    classifier = pipeline.getStages()[-1]

    if model_type == "rf":
        param_grid = (
            ParamGridBuilder()
            .addGrid(classifier.numTrees,  [100, 200])
            .addGrid(classifier.maxDepth,  [6, 10])
            .build()
        )
    else:
        param_grid = (
            ParamGridBuilder()
            .addGrid(classifier.maxIter,  [50, 100])
            .addGrid(classifier.maxDepth, [4, 6])
            .build()
        )

    cv = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=param_grid,
        evaluator=evaluator,
        numFolds=num_folds,
        seed=42,
        parallelism=2,
    )

    logger.info("Starting %d-fold cross-validation …", num_folds)
    cv_model = cv.fit(train_df)
    logger.info("CV complete. Best AUC: %.4f", max(cv_model.avgMetrics))
    return cv_model.bestModel


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_model(model, test_df: DataFrame, output_path: str) -> dict:
    predictions = model.transform(test_df)

    auc_eval = BinaryClassificationEvaluator(
        labelCol=LABEL_COL, metricName="areaUnderROC"
    )
    pr_eval = BinaryClassificationEvaluator(
        labelCol=LABEL_COL, metricName="areaUnderPR"
    )
    acc_eval = MulticlassClassificationEvaluator(
        labelCol=LABEL_COL, predictionCol="prediction", metricName="accuracy"
    )
    f1_eval = MulticlassClassificationEvaluator(
        labelCol=LABEL_COL, predictionCol="prediction", metricName="f1"
    )
    prec_eval = MulticlassClassificationEvaluator(
        labelCol=LABEL_COL, predictionCol="prediction", metricName="weightedPrecision"
    )
    rec_eval = MulticlassClassificationEvaluator(
        labelCol=LABEL_COL, predictionCol="prediction", metricName="weightedRecall"
    )

    metrics = {
        "auc_roc":   round(auc_eval.evaluate(predictions), 4),
        "auc_pr":    round(pr_eval.evaluate(predictions),  4),
        "accuracy":  round(acc_eval.evaluate(predictions), 4),
        "f1_score":  round(f1_eval.evaluate(predictions),  4),
        "precision": round(prec_eval.evaluate(predictions),4),
        "recall":    round(rec_eval.evaluate(predictions), 4),
    }

    logger.info("Model Metrics: %s", json.dumps(metrics, indent=2))

    # Confusion matrix
    conf_matrix = (
        predictions
        .groupBy(LABEL_COL, "prediction")
        .count()
        .orderBy(LABEL_COL, "prediction")
    )
    logger.info("Confusion Matrix:")
    conf_matrix.show()

    # Feature importances (RF only)
    rf_stage = _get_rf_stage(model)
    if rf_stage is not None:
        feature_names = (
            [f"{c}_imp" for c in NUMERIC_FEATURES] +
            [f"{c}_idx" for c in CATEGORICAL_FEATURES]
        )
        importances = sorted(
            zip(feature_names, rf_stage.featureImportances.toArray()),
            key=lambda x: -x[1],
        )
        logger.info("Top 10 Feature Importances:")
        for name, imp in importances[:10]:
            logger.info("  %-45s  %.4f", name, imp)

    # Persist metrics
    Path(output_path).mkdir(parents=True, exist_ok=True)
    with open(f"{output_path}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def _get_rf_stage(model):
    try:
        from pyspark.ml.classification import RandomForestClassificationModel
        for stage in model.stages:
            if isinstance(stage, RandomForestClassificationModel):
                return stage
    except Exception:
        pass
    return None


# ─── Entry Point ──────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="MediRisk Fraud Detection Training")
    parser.add_argument("--gold-dir",     required=True)
    parser.add_argument("--model-dir",    required=True)
    parser.add_argument("--metrics-dir",  required=True)
    parser.add_argument("--model-type",   default="rf", choices=["rf", "gbt"])
    parser.add_argument("--cv-folds",     type=int, default=3)
    args = parser.parse_args()

    spark = get_spark()

    train_df, test_df = load_and_prepare(spark, args.gold_dir)

    pipeline = build_pipeline(args.model_type)
    model    = train_with_cv(pipeline, train_df, args.model_type, args.cv_folds)

    metrics = evaluate_model(model, test_df, args.metrics_dir)

    model_path = f"{args.model_dir}/fraud_detection_{args.model_type}"
    model.write().overwrite().save(model_path)
    logger.info("Model saved to %s", model_path)

    spark.stop()
    logger.info("✅ Fraud detection training complete. AUC-ROC: %.4f", metrics["auc_roc"])


if __name__ == "__main__":
    main()
