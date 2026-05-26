"""
MediRisk Intelligence Platform
Batch Scoring — Apply Trained Fraud Detection Model to New Claims
Loads the saved MLlib model and scores the full Gold feature set,
writing fraud probabilities back to Delta and exporting to S3 for Snowpipe.
"""

from __future__ import annotations

import logging
from pathlib import Path

from pyspark.ml import PipelineModel
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def get_spark(app_name: str = "MediRisk-BatchScoring") -> SparkSession:
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


# ─── Extract fraud probability from Vector column ─────────────────────────────

def extract_fraud_prob(df: DataFrame) -> DataFrame:
    """Extract P(fraud=1) from the probability vector column."""
    extract_prob = F.udf(lambda v: float(v[1]) if v is not None else None, DoubleType())
    return df.withColumn("fraud_probability", extract_prob(F.col("probability")))


# ─── Score ────────────────────────────────────────────────────────────────────

def score_claims(
    spark:       SparkSession,
    gold_path:   str,
    model_path:  str,
    output_path: str,
    threshold:   float = 0.5,
) -> None:

    logger.info("Loading model from %s", model_path)
    model = PipelineModel.load(model_path)

    logger.info("Loading Gold claims features …")
    features_df = spark.read.format("delta").load(f"{gold_path}/claims_features")

    logger.info("Scoring %d claims …", features_df.count())
    scored = model.transform(features_df)
    scored = extract_fraud_prob(scored)

    # Final output schema
    output = (
        scored
        .select(
            "claim_id",
            "patient_id",
            "provider_id",
            "service_date",
            "billed_amount",
            "diagnosis_code",
            "procedure_code",
            "fraud_probability",
            F.col("prediction").alias("predicted_fraud_label"),
            F.when(F.col("fraud_probability") >= threshold, F.lit(1))
             .otherwise(F.lit(0)).alias("fraud_flag"),
            F.when(F.col("fraud_probability") >= 0.8, F.lit("HIGH"))
             .when(F.col("fraud_probability") >= 0.5, F.lit("MEDIUM"))
             .when(F.col("fraud_probability") >= 0.3, F.lit("LOW"))
             .otherwise(F.lit("CLEAR")).alias("risk_band"),
        )
        .withColumn("scored_at", F.current_timestamp())
    )

    # Stats
    high   = output.filter(F.col("risk_band") == "HIGH").count()
    medium = output.filter(F.col("risk_band") == "MEDIUM").count()
    low    = output.filter(F.col("risk_band") == "LOW").count()
    clear  = output.filter(F.col("risk_band") == "CLEAR").count()
    total  = output.count()

    logger.info(
        "Scoring summary — Total: %d  |  HIGH: %d  |  MEDIUM: %d  |  LOW: %d  |  CLEAR: %d",
        total, high, medium, low, clear,
    )

    # Write to Delta for internal use
    (
        output.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .partitionBy("risk_band")
        .save(f"{output_path}/fraud_scores_delta")
    )
    logger.info("Scores written to Delta → %s/fraud_scores_delta", output_path)

    # Write to Parquet for Snowpipe auto-ingest
    (
        output.write
        .format("parquet")
        .mode("overwrite")
        .save(f"{output_path}/fraud_scores_snowpipe")
    )
    logger.info("Scores written to Parquet (Snowpipe) → %s/fraud_scores_snowpipe", output_path)


# ─── Entry Point ──────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="MediRisk Batch Scoring")
    parser.add_argument("--gold-dir",    required=True)
    parser.add_argument("--model-path",  required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--threshold",   type=float, default=0.5)
    args = parser.parse_args()

    spark = get_spark()
    score_claims(spark, args.gold_dir, args.model_path, args.output_path, args.threshold)
    spark.stop()
    logger.info("✅ Batch scoring complete.")


if __name__ == "__main__":
    main()
