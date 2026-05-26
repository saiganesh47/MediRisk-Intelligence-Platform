"""
MediRisk Intelligence Platform
Gold Layer — Feature Engineering
Joins Silver tables, applies advanced window functions, and produces
the ML-ready feature store for fraud detection and patient risk scoring.
"""

from __future__ import annotations

import logging

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import Window

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ─── Session ──────────────────────────────────────────────────────────────────

def get_spark(app_name: str = "MediRisk-Gold") -> SparkSession:
    return (
        SparkSession.builder
        .appName(app_name)
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        .config("spark.sql.adaptive.enabled",                   "true")
        .config("spark.sql.adaptive.skewJoin.enabled",          "true")
        .config("spark.sql.shuffle.partitions",                 "400")
        .getOrCreate()
    )


# ─── Window Specs ─────────────────────────────────────────────────────────────

def _provider_90d_window():
    """Rolling 90-day window per provider ordered by service_date."""
    return (
        Window
        .partitionBy("provider_id")
        .orderBy(F.col("service_date").cast("long"))
        .rangeBetween(-90 * 86_400, 0)   # 90 days in seconds
    )


def _provider_all_time():
    return Window.partitionBy("provider_id")


def _patient_all_time():
    return Window.partitionBy("patient_id")


def _patient_30d_window():
    return (
        Window
        .partitionBy("patient_id")
        .orderBy(F.col("service_date").cast("long"))
        .rangeBetween(-30 * 86_400, 0)
    )


# ─── Claims Feature Engineering ───────────────────────────────────────────────

def build_claims_features(claims: DataFrame) -> DataFrame:
    """
    Adds provider-level and patient-level rolling aggregation features
    to every claim record. These become the ML model's input features.
    """
    logger.info("Building claims feature set …")

    p90  = _provider_90d_window()
    pall = _provider_all_time()
    pt30 = _patient_30d_window()
    ptall= _patient_all_time()

    df = (
        claims

        # ── Provider rolling 90-day features ──────────────────────────────
        .withColumn("prov_90d_claim_count",
                    F.count("claim_id").over(p90))
        .withColumn("prov_90d_billed_total",
                    F.sum("billed_amount").over(p90))
        .withColumn("prov_90d_avg_billed",
                    F.avg("billed_amount").over(p90))
        .withColumn("prov_90d_stddev_billed",
                    F.stddev("billed_amount").over(p90))
        .withColumn("prov_90d_unique_patients",
                    F.approx_count_distinct("patient_id").over(p90))
        .withColumn("prov_90d_fraud_count",
                    F.sum(F.col("is_fraud").cast("int")).over(p90))

        # ── Provider all-time features ─────────────────────────────────────
        .withColumn("prov_total_claims",
                    F.count("claim_id").over(pall))
        .withColumn("prov_total_billed",
                    F.sum("billed_amount").over(pall))
        .withColumn("prov_fraud_rate",
                    F.round(
                        F.sum(F.col("is_fraud").cast("int")).over(pall) /
                        F.count("claim_id").over(pall),
                        6,
                    ))
        .withColumn("prov_avg_lag_days",
                    F.avg("lag_days").over(pall))

        # ── Patient rolling 30-day features ───────────────────────────────
        .withColumn("pat_30d_claim_count",
                    F.count("claim_id").over(pt30))
        .withColumn("pat_30d_billed_total",
                    F.sum("billed_amount").over(pt30))
        .withColumn("pat_30d_unique_providers",
                    F.approx_count_distinct("provider_id").over(pt30))

        # ── Patient all-time features ──────────────────────────────────────
        .withColumn("pat_total_claims",
                    F.count("claim_id").over(ptall))
        .withColumn("pat_total_billed",
                    F.sum("billed_amount").over(ptall))
        .withColumn("pat_avg_billed",
                    F.avg("billed_amount").over(ptall))

        # ── Claim-level derived features ───────────────────────────────────
        .withColumn(
            "billed_vs_provider_avg",
            F.round(F.col("billed_amount") / (F.col("prov_90d_avg_billed") + 1), 4),
        )
        .withColumn(
            "is_high_lag",
            (F.col("lag_days") > 30).cast("int"),
        )
        .withColumn(
            "is_weekend_service",
            (F.dayofweek("service_date").isin([1, 7])).cast("int"),
        )
        .withColumn(
            "fraud_signals_count",
            F.size(F.from_json("fraud_signals",
                               "array<string>")),
        )
    )

    return df


# ─── Patient Risk Score (Gold Aggregate) ──────────────────────────────────────

def build_patient_risk_scores(
    claims:       DataFrame,
    patients:     DataFrame,
    transactions: DataFrame,
) -> DataFrame:
    """
    Aggregate patient-level risk features into a single Gold record per patient.
    This feeds the Snowflake patient_risk_summary Dynamic Table.
    """
    logger.info("Building patient risk scores …")

    # Claims aggregation per patient
    claims_agg = (
        claims
        .groupBy("patient_id")
        .agg(
            F.count("claim_id").alias("total_claims"),
            F.sum("billed_amount").alias("lifetime_billed"),
            F.avg("billed_amount").alias("avg_claim_billed"),
            F.max("billed_amount").alias("max_claim_billed"),
            F.sum("paid_amount").alias("lifetime_paid"),
            F.sum(F.col("is_fraud").cast("int")).alias("fraud_claim_count"),
            F.countDistinct("provider_id").alias("distinct_providers"),
            F.countDistinct("procedure_code").alias("distinct_procedures"),
            F.avg("lag_days").alias("avg_submission_lag"),
            F.max("service_date").alias("last_service_date"),
        )
        .withColumn(
            "fraud_claim_rate",
            F.round(F.col("fraud_claim_count") / F.col("total_claims"), 6),
        )
    )

    # Financial transactions aggregation per patient
    tx_agg = (
        transactions
        .groupBy("patient_id")
        .agg(
            F.count("transaction_id").alias("total_transactions"),
            F.sum("amount").alias("total_tx_amount"),
            F.avg("amount").alias("avg_tx_amount"),
            F.max("amount").alias("max_tx_amount"),
            F.sum(F.col("is_anomalous").cast("int")).alias("anomalous_tx_count"),
        )
        .withColumn(
            "anomalous_tx_rate",
            F.round(F.col("anomalous_tx_count") / F.col("total_transactions"), 6),
        )
    )

    # Join everything
    risk = (
        patients
        .join(claims_agg,  "patient_id", "left")
        .join(tx_agg,      "patient_id", "left")
        .fillna(0, subset=[
            "total_claims", "lifetime_billed", "fraud_claim_count",
            "total_transactions", "anomalous_tx_count",
        ])
    )

    # Composite risk score (rule-based pre-ML score, 0–100)
    risk = risk.withColumn(
        "composite_risk_score",
        F.least(
            F.lit(100.0),
            F.round(
                (F.col("fraud_claim_rate")       * 40) +
                (F.col("anomalous_tx_rate")      * 25) +
                (F.col("chronic_conditions")     *  5) +
                (F.col("distinct_providers") / F.col("total_claims").cast("double") * 20) +
                (F.when(F.col("avg_submission_lag") > 30, 10).otherwise(0)),
                2,
            ),
        ),
    ).withColumn(
        "risk_tier",
        F.when(F.col("composite_risk_score") >= 70, F.lit("HIGH"))
         .when(F.col("composite_risk_score") >= 40, F.lit("MEDIUM"))
         .otherwise(F.lit("LOW")),
    ).withColumn(
        "score_computed_at", F.current_timestamp(),
    )

    return risk


# ─── Provider Risk Summary ────────────────────────────────────────────────────

def build_provider_risk_summary(claims: DataFrame, providers: DataFrame) -> DataFrame:
    logger.info("Building provider risk summary …")

    claims_agg = (
        claims
        .groupBy("provider_id")
        .agg(
            F.count("claim_id").alias("total_claims"),
            F.sum("billed_amount").alias("total_billed"),
            F.avg("billed_amount").alias("avg_billed"),
            F.sum(F.col("is_fraud").cast("int")).alias("fraud_count"),
            F.countDistinct("patient_id").alias("unique_patients"),
            F.countDistinct("diagnosis_code").alias("unique_diagnoses"),
            F.avg("lag_days").alias("avg_lag_days"),
            F.max("service_date").alias("last_service_date"),
        )
        .withColumn(
            "fraud_rate",
            F.round(F.col("fraud_count") / F.col("total_claims"), 6),
        )
    )

    return (
        providers
        .join(claims_agg, "provider_id", "left")
        .fillna(0, subset=["total_claims", "fraud_count"])
        .withColumn(
            "provider_risk_score",
            F.least(
                F.lit(100.0),
                F.round(
                    (F.col("fraud_rate")      * 50) +
                    (F.when(~F.col("license_valid"), 30).otherwise(0)) +
                    (F.when(F.col("is_flagged"),     20).otherwise(0)),
                    2,
                ),
            ),
        )
        .withColumn("score_computed_at", F.current_timestamp())
    )


# ─── Writers ──────────────────────────────────────────────────────────────────

def _write_gold(df: DataFrame, path: str, partition_cols: list[str] | None = None) -> None:
    writer = (
        df.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
    )
    if partition_cols:
        writer = writer.partitionBy(*partition_cols)
    writer.save(path)
    logger.info("Gold table written → %s  (%d rows)", path, df.count())


# ─── Entry Point ──────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="MediRisk Gold Feature Engineering")
    parser.add_argument("--silver-dir", required=True)
    parser.add_argument("--gold-dir",   required=True)
    args = parser.parse_args()

    spark = get_spark()

    logger.info("Reading Silver tables …")
    claims       = spark.read.format("delta").load(f"{args.silver_dir}/claims")
    patients     = spark.read.format("delta").load(f"{args.silver_dir}/patients")
    providers    = spark.read.format("delta").load(f"{args.silver_dir}/providers")
    transactions = spark.read.format("delta").load(f"{args.silver_dir}/transactions")

    # Cache hot tables
    claims.cache()
    patients.cache()

    # Build Gold tables
    claims_features = build_claims_features(claims)
    _write_gold(claims_features, f"{args.gold_dir}/claims_features", ["service_year"])

    patient_risk = build_patient_risk_scores(claims_features, patients, transactions)
    _write_gold(patient_risk, f"{args.gold_dir}/patient_risk_scores")

    provider_risk = build_provider_risk_summary(claims_features, providers)
    _write_gold(provider_risk, f"{args.gold_dir}/provider_risk_summary")

    spark.stop()
    logger.info("✅ Gold feature engineering complete.")


if __name__ == "__main__":
    main()
