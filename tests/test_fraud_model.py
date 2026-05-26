"""
MediRisk Intelligence Platform
Tests — Fraud Detection Model
Validates data preparation, pipeline construction,
prediction output schema, and model evaluation thresholds.
Run with: pytest tests/test_fraud_model.py -v
"""

from __future__ import annotations

import pytest
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import Imputer, StringIndexer, VectorAssembler
from pyspark.sql import Row, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    BooleanType, DoubleType, StringType, StructField, StructType,
)

from pyspark.ml.evaluation import BinaryClassificationEvaluator


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def spark():
    session = (
        SparkSession.builder
        .master("local[2]")
        .appName("MediRisk-FraudModel-Tests")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()


@pytest.fixture
def sample_features_df(spark):
    """
    A small labelled dataset: 60 non-fraud, 40 fraud, with distinguishable features.
    """
    rows = []
    for i in range(60):
        rows.append(Row(
            claim_id=f"nf{i:03d}",
            billed_amount=float(100 + i * 5),
            allowed_amount=float(90 + i * 4),
            paid_amount=float(85 + i * 4),
            lag_days=int(2 + i % 10),
            billed_to_allowed_ratio=float(1.1 + i * 0.001),
            billed_vs_provider_avg=float(0.9 + i * 0.001),
            prov_90d_claim_count=float(10 + i % 20),
            prov_90d_billed_total=float(5000.0),
            prov_90d_avg_billed=float(500.0),
            prov_90d_stddev_billed=float(50.0),
            prov_90d_unique_patients=float(8.0),
            prov_90d_fraud_count=float(0.0),
            prov_total_claims=float(200.0),
            prov_fraud_rate=float(0.01),
            prov_avg_lag_days=float(5.0),
            pat_30d_claim_count=float(1.0),
            pat_30d_billed_total=float(500.0),
            pat_30d_unique_providers=float(1.0),
            pat_total_claims=float(5.0),
            pat_avg_billed=float(300.0),
            fraud_signals_count=int(0),
            is_high_lag=int(0),
            is_weekend_service=int(0),
            diagnosis_code="I10",
            procedure_code="99213",
            claim_status="APPROVED",
            label=0.0,
        ))

    for i in range(40):
        rows.append(Row(
            claim_id=f"fr{i:03d}",
            billed_amount=float(5000 + i * 100),
            allowed_amount=float(500 + i * 10),
            paid_amount=float(480 + i * 10),
            lag_days=int(35 + i % 10),
            billed_to_allowed_ratio=float(8.0 + i * 0.1),
            billed_vs_provider_avg=float(5.0 + i * 0.1),
            prov_90d_claim_count=float(200 + i),
            prov_90d_billed_total=float(500_000.0),
            prov_90d_avg_billed=float(2500.0),
            prov_90d_stddev_billed=float(1000.0),
            prov_90d_unique_patients=float(2.0),
            prov_90d_fraud_count=float(50.0),
            prov_total_claims=float(500.0),
            prov_fraud_rate=float(0.25),
            prov_avg_lag_days=float(40.0),
            pat_30d_claim_count=float(10.0),
            pat_30d_billed_total=float(50_000.0),
            pat_30d_unique_providers=float(8.0),
            pat_total_claims=float(50.0),
            pat_avg_billed=float(3000.0),
            fraud_signals_count=int(3),
            is_high_lag=int(1),
            is_weekend_service=int(1),
            diagnosis_code="E11.9",
            procedure_code="99215",
            claim_status="PENDING",
            label=1.0,
        ))

    schema = StructType([
        StructField("claim_id",               StringType(),  False),
        StructField("billed_amount",          DoubleType(),  True),
        StructField("allowed_amount",         DoubleType(),  True),
        StructField("paid_amount",            DoubleType(),  True),
        StructField("lag_days",               DoubleType(),  True),
        StructField("billed_to_allowed_ratio",DoubleType(),  True),
        StructField("billed_vs_provider_avg", DoubleType(),  True),
        StructField("prov_90d_claim_count",   DoubleType(),  True),
        StructField("prov_90d_billed_total",  DoubleType(),  True),
        StructField("prov_90d_avg_billed",    DoubleType(),  True),
        StructField("prov_90d_stddev_billed", DoubleType(),  True),
        StructField("prov_90d_unique_patients",DoubleType(), True),
        StructField("prov_90d_fraud_count",   DoubleType(),  True),
        StructField("prov_total_claims",      DoubleType(),  True),
        StructField("prov_fraud_rate",        DoubleType(),  True),
        StructField("prov_avg_lag_days",      DoubleType(),  True),
        StructField("pat_30d_claim_count",    DoubleType(),  True),
        StructField("pat_30d_billed_total",   DoubleType(),  True),
        StructField("pat_30d_unique_providers",DoubleType(), True),
        StructField("pat_total_claims",       DoubleType(),  True),
        StructField("pat_avg_billed",         DoubleType(),  True),
        StructField("fraud_signals_count",    DoubleType(),  True),
        StructField("is_high_lag",            DoubleType(),  True),
        StructField("is_weekend_service",     DoubleType(),  True),
        StructField("diagnosis_code",         StringType(),  True),
        StructField("procedure_code",         StringType(),  True),
        StructField("claim_status",           StringType(),  True),
        StructField("label",                  DoubleType(),  False),
    ])

    return spark.createDataFrame(rows, schema)


# ─── Pipeline Construction Tests ──────────────────────────────────────────────

class TestPipelineConstruction:

    def test_pipeline_has_correct_stage_count(self, sample_features_df):
        """Pipeline should have: 3 indexers + imputer + assembler + scaler + classifier = 7"""
        from pyspark.ml.feature import MinMaxScaler

        numeric_cols     = ["billed_amount", "lag_days", "prov_fraud_rate"]
        categorical_cols = ["diagnosis_code", "procedure_code", "claim_status"]

        indexers  = [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
                     for c in categorical_cols]
        imputer   = Imputer(inputCols=numeric_cols,
                            outputCols=[f"{c}_imp" for c in numeric_cols], strategy="median")
        assembler = VectorAssembler(
            inputCols=[f"{c}_imp" for c in numeric_cols] + [f"{c}_idx" for c in categorical_cols],
            outputCol="raw_features", handleInvalid="keep",
        )
        scaler    = MinMaxScaler(inputCol="raw_features", outputCol="features")
        clf       = RandomForestClassifier(labelCol="label", featuresCol="features",
                                           numTrees=10, maxDepth=3, seed=42)

        pipeline = Pipeline(stages=indexers + [imputer, assembler, scaler, clf])
        assert len(pipeline.getStages()) == 7

    def test_pipeline_fits_and_transforms(self, spark, sample_features_df):
        from pyspark.ml.feature import MinMaxScaler

        numeric_cols     = [
            "billed_amount", "lag_days", "prov_fraud_rate", "fraud_signals_count",
            "billed_to_allowed_ratio", "prov_90d_claim_count",
        ]
        categorical_cols = ["diagnosis_code", "procedure_code", "claim_status"]

        indexers  = [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
                     for c in categorical_cols]
        imputer   = Imputer(inputCols=numeric_cols,
                            outputCols=[f"{c}_imp" for c in numeric_cols], strategy="median")
        assembler = VectorAssembler(
            inputCols=[f"{c}_imp" for c in numeric_cols] + [f"{c}_idx" for c in categorical_cols],
            outputCol="raw_features", handleInvalid="keep",
        )
        scaler    = MinMaxScaler(inputCol="raw_features", outputCol="features")
        clf       = RandomForestClassifier(labelCol="label", featuresCol="features",
                                           numTrees=10, maxDepth=3, seed=42)

        pipeline = Pipeline(stages=indexers + [imputer, assembler, scaler, clf])
        model    = pipeline.fit(sample_features_df)
        preds    = model.transform(sample_features_df)

        assert "prediction"   in preds.columns
        assert "probability"  in preds.columns
        assert "rawPrediction" in preds.columns
        assert preds.count()  == sample_features_df.count()


# ─── Data Balance Tests ───────────────────────────────────────────────────────

class TestDataBalance:

    def test_class_imbalance_ratio(self, sample_features_df):
        total  = sample_features_df.count()
        fraud  = sample_features_df.filter(F.col("label") == 1).count()
        ratio  = fraud / total
        assert 0.20 <= ratio <= 0.80, f"Class imbalance too severe: {ratio:.2%}"

    def test_train_test_split_sizes(self, sample_features_df):
        train, test = sample_features_df.randomSplit([0.8, 0.2], seed=42)
        total       = sample_features_df.count()
        assert abs(train.count() / total - 0.8) < 0.1
        assert abs(test.count()  / total - 0.2) < 0.1

    def test_both_classes_in_train_and_test(self, sample_features_df):
        train, test = sample_features_df.randomSplit([0.8, 0.2], seed=42)
        for split_name, split_df in [("train", train), ("test", test)]:
            labels = {r["label"] for r in split_df.select("label").distinct().collect()}
            assert 0.0 in labels, f"{split_name} missing class 0"
            assert 1.0 in labels, f"{split_name} missing class 1"


# ─── Model Quality Tests ──────────────────────────────────────────────────────

class TestModelQuality:

    @pytest.fixture(scope="class")
    def trained_model(self, spark, sample_features_df):
        from pyspark.ml.feature import MinMaxScaler

        numeric_cols     = [
            "billed_amount", "lag_days", "prov_fraud_rate", "fraud_signals_count",
            "billed_to_allowed_ratio", "prov_90d_fraud_count", "is_high_lag",
        ]
        categorical_cols = ["diagnosis_code", "procedure_code", "claim_status"]

        indexers  = [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
                     for c in categorical_cols]
        imputer   = Imputer(inputCols=numeric_cols,
                            outputCols=[f"{c}_imp" for c in numeric_cols], strategy="median")
        assembler = VectorAssembler(
            inputCols=[f"{c}_imp" for c in numeric_cols] + [f"{c}_idx" for c in categorical_cols],
            outputCol="raw_features", handleInvalid="keep",
        )
        scaler    = MinMaxScaler(inputCol="raw_features", outputCol="features")
        clf       = RandomForestClassifier(labelCol="label", featuresCol="features",
                                           numTrees=50, maxDepth=5, seed=42)

        pipeline = Pipeline(stages=indexers + [imputer, assembler, scaler, clf])
        return pipeline.fit(sample_features_df)

    def test_auc_roc_above_threshold(self, trained_model, sample_features_df):
        preds    = trained_model.transform(sample_features_df)
        evaluator = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")
        auc      = evaluator.evaluate(preds)
        assert auc >= 0.85, f"AUC-ROC too low: {auc:.4f} (expected ≥ 0.85)"

    def test_predictions_are_binary(self, trained_model, sample_features_df):
        preds  = trained_model.transform(sample_features_df)
        labels = {r["prediction"] for r in preds.select("prediction").distinct().collect()}
        assert labels.issubset({0.0, 1.0}), f"Unexpected prediction values: {labels}"

    def test_fraud_probability_in_range(self, trained_model, sample_features_df):
        from pyspark.sql.types import DoubleType
        preds = trained_model.transform(sample_features_df)

        extract = F.udf(lambda v: float(v[1]) if v else None, DoubleType())
        probs   = preds.withColumn("fraud_prob", extract(F.col("probability")))

        out_of_range = probs.filter(
            (F.col("fraud_prob") < 0) | (F.col("fraud_prob") > 1)
        ).count()
        assert out_of_range == 0

    def test_no_null_predictions(self, trained_model, sample_features_df):
        preds     = trained_model.transform(sample_features_df)
        null_preds = preds.filter(F.col("prediction").isNull()).count()
        assert null_preds == 0


# ─── Scoring Output Tests ─────────────────────────────────────────────────────

class TestScoringOutput:

    def test_risk_band_assignment_coverage(self, spark):
        schema = StructType([
            StructField("fraud_prob", DoubleType(), True),
        ])
        df = spark.createDataFrame([
            Row(fraud_prob=0.9),
            Row(fraud_prob=0.6),
            Row(fraud_prob=0.35),
            Row(fraud_prob=0.1),
        ], schema)

        df = df.withColumn(
            "risk_band",
            F.when(F.col("fraud_prob") >= 0.8,  F.lit("HIGH"))
             .when(F.col("fraud_prob") >= 0.5,  F.lit("MEDIUM"))
             .when(F.col("fraud_prob") >= 0.3,  F.lit("LOW"))
             .otherwise(F.lit("CLEAR")),
        )

        bands = {row["fraud_prob"]: row["risk_band"] for row in df.collect()}
        assert bands[0.9]  == "HIGH"
        assert bands[0.6]  == "MEDIUM"
        assert bands[0.35] == "LOW"
        assert bands[0.1]  == "CLEAR"

    def test_fraud_flag_matches_threshold(self, spark):
        schema = StructType([StructField("prob", DoubleType(), True)])
        df = spark.createDataFrame([
            Row(prob=0.6), Row(prob=0.4), Row(prob=0.5),
        ], schema)

        threshold = 0.5
        df = df.withColumn(
            "fraud_flag",
            F.when(F.col("prob") >= threshold, F.lit(1)).otherwise(F.lit(0)),
        )

        flags = {row["prob"]: row["fraud_flag"] for row in df.collect()}
        assert flags[0.6] == 1
        assert flags[0.5] == 1
        assert flags[0.4] == 0
