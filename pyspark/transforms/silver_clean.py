"""
MediRisk Intelligence Platform
Silver Layer — Cleansing, Standardisation & Deduplication
Reads Bronze Delta tables, applies business rules, and writes
clean, conformed Silver tables.
"""

from __future__ import annotations

import logging

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import Window
from pyspark.sql.types import DateType, DoubleType

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ─── Session ──────────────────────────────────────────────────────────────────

def get_spark(app_name: str = "MediRisk-Silver") -> SparkSession:
    return (
        SparkSession.builder
        .appName(app_name)
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        .config("spark.sql.adaptive.enabled",               "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .getOrCreate()
    )


# ─── Claims Silver ────────────────────────────────────────────────────────────

def transform_claims(spark: SparkSession, bronze_path: str, silver_path: str) -> None:
    logger.info("Transforming claims Bronze → Silver")

    df = spark.read.format("delta").load(f"{bronze_path}/claims")

    # 1. Drop quarantined rows
    df = df.filter(F.col("_dq_status") == "PASS")

    # 2. Cast and clean dates
    df = (
        df
        .withColumn("service_date",   F.to_date("service_date"))
        .withColumn("submitted_date", F.to_date("submitted_date"))
        .withColumn("ingestion_ts",   F.to_timestamp("ingestion_ts"))
    )

    # 3. Standardise text fields
    df = (
        df
        .withColumn("claim_status",   F.upper(F.trim("claim_status")))
        .withColumn("diagnosis_code", F.upper(F.trim("diagnosis_code")))
        .withColumn("procedure_code", F.upper(F.trim("procedure_code")))
    )

    # 4. Numeric guards — billed must be ≥ allowed ≥ 0
    df = (
        df
        .withColumn(
            "billed_amount",
            F.when(F.col("billed_amount") < 0, F.lit(None).cast(DoubleType()))
             .otherwise(F.col("billed_amount")),
        )
        .withColumn(
            "allowed_amount",
            F.when(F.col("allowed_amount") > F.col("billed_amount"), F.col("billed_amount"))
             .when(F.col("allowed_amount") < 0, F.lit(None).cast(DoubleType()))
             .otherwise(F.col("allowed_amount")),
        )
        .withColumn(
            "paid_amount",
            F.when(F.col("paid_amount") > F.col("allowed_amount"), F.col("allowed_amount"))
             .when(F.col("paid_amount") < 0, F.lit(0.0))
             .otherwise(F.col("paid_amount")),
        )
    )

    # 5. Derived columns
    df = (
        df
        .withColumn(
            "lag_days",  # submission delay
            F.datediff("submitted_date", "service_date"),
        )
        .withColumn(
            "billed_to_allowed_ratio",
            F.round(F.col("billed_amount") / F.col("allowed_amount"), 4),
        )
        .withColumn(
            "service_year",  F.year("service_date"),
        )
        .withColumn(
            "service_quarter",
            F.concat(F.year("service_date"), F.lit("-Q"), F.quarter("service_date")),
        )
    )

    # 6. Deduplication — keep latest submission per claim_id
    win = Window.partitionBy("claim_id").orderBy(F.col("submitted_date").desc())
    df = (
        df
        .withColumn("_row_num", F.row_number().over(win))
        .filter(F.col("_row_num") == 1)
        .drop("_row_num")
    )

    # 7. Drop raw Bronze metadata before Silver write
    silver_cols = [c for c in df.columns if not c.startswith("_")]
    df = df.select(*silver_cols)

    _write_silver(df, f"{silver_path}/claims", ["service_year"])
    _log_summary("claims_silver", df)


# ─── Patients Silver ──────────────────────────────────────────────────────────

def transform_patients(spark: SparkSession, bronze_path: str, silver_path: str) -> None:
    logger.info("Transforming patients Bronze → Silver")

    df = spark.read.format("delta").load(f"{bronze_path}/patients")
    df = df.filter(F.col("_dq_status") == "PASS")

    df = (
        df
        .withColumn("dob",        F.to_date("dob"))
        .withColumn("created_at", F.to_timestamp("created_at"))
        .withColumn("gender",     F.upper(F.trim("gender")))
        .withColumn("state",      F.upper(F.trim("state")))
        .withColumn(
            "age",
            F.floor(F.months_between(F.current_date(), F.col("dob")) / 12).cast("int"),
        )
        .withColumn(
            "age_band",
            F.when(F.col("age") < 30,  F.lit("18-29"))
             .when(F.col("age") < 45,  F.lit("30-44"))
             .when(F.col("age") < 60,  F.lit("45-59"))
             .when(F.col("age") < 75,  F.lit("60-74"))
             .otherwise(F.lit("75+")),
        )
        .withColumn(
            "high_risk_flag",
            (F.col("chronic_conditions") >= 3).cast("boolean"),
        )
    )

    # Dedup on patient_id — keep most-recently-created
    win = Window.partitionBy("patient_id").orderBy(F.col("created_at").desc())
    df = (
        df
        .withColumn("_rn", F.row_number().over(win))
        .filter(F.col("_rn") == 1)
        .drop("_rn")
    )

    silver_cols = [c for c in df.columns if not c.startswith("_")]
    df = df.select(*silver_cols)

    _write_silver(df, f"{silver_path}/patients", ["state"])
    _log_summary("patients_silver", df)


# ─── Providers Silver ─────────────────────────────────────────────────────────

def transform_providers(spark: SparkSession, bronze_path: str, silver_path: str) -> None:
    logger.info("Transforming providers Bronze → Silver")

    df = spark.read.format("delta").load(f"{bronze_path}/providers")
    df = df.filter(F.col("_dq_status") == "PASS")

    df = (
        df
        .withColumn("joined_date", F.to_date("joined_date"))
        .withColumn("state",       F.upper(F.trim("state")))
        .withColumn("specialty",   F.initcap(F.trim("specialty")))
        .withColumn(
            "years_active",
            F.floor(F.months_between(F.current_date(), F.col("joined_date")) / 12).cast("int"),
        )
        .withColumn(
            "risk_tier",
            F.when(F.col("is_flagged") & ~F.col("license_valid"), F.lit("HIGH"))
             .when(F.col("is_flagged"),                            F.lit("MEDIUM"))
             .otherwise(F.lit("LOW")),
        )
    )

    silver_cols = [c for c in df.columns if not c.startswith("_")]
    df = df.select(*silver_cols)

    _write_silver(df, f"{silver_path}/providers", ["state"])
    _log_summary("providers_silver", df)


# ─── Transactions Silver ──────────────────────────────────────────────────────

def transform_transactions(spark: SparkSession, bronze_path: str, silver_path: str) -> None:
    logger.info("Transforming transactions Bronze → Silver")

    df = spark.read.format("delta").load(f"{bronze_path}/transactions")
    df = df.filter(F.col("_dq_status") == "PASS")

    df = (
        df
        .withColumn("transaction_date", F.to_date("transaction_date"))
        .withColumn("ingestion_ts",      F.to_timestamp("ingestion_ts"))
        .withColumn("transaction_type",  F.upper(F.trim("transaction_type")))
        .withColumn("channel",           F.upper(F.trim("channel")))
        .withColumn(
            "amount",
            F.when(F.col("amount") < 0, F.abs("amount")).otherwise(F.col("amount")),
        )
        .withColumn("tx_year",  F.year("transaction_date"))
        .withColumn("tx_month", F.month("transaction_date"))
    )

    silver_cols = [c for c in df.columns if not c.startswith("_")]
    df = df.select(*silver_cols)

    _write_silver(df, f"{silver_path}/transactions", ["tx_year"])
    _log_summary("transactions_silver", df)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _write_silver(df: DataFrame, path: str, partition_cols: list[str]) -> None:
    (
        df.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .partitionBy(*partition_cols)
        .save(path)
    )
    logger.info("Silver table written → %s", path)


def _log_summary(name: str, df: DataFrame) -> None:
    count = df.count()
    logger.info("[%s] rows: %d", name, count)


# ─── Entry Point ──────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="MediRisk Silver Transforms")
    parser.add_argument("--bronze-dir", required=True)
    parser.add_argument("--silver-dir", required=True)
    parser.add_argument("--entity",     default="all",
                        help="claims|patients|providers|transactions|all")
    args = parser.parse_args()

    spark = get_spark()

    run = {
        "claims":       transform_claims,
        "patients":     transform_patients,
        "providers":    transform_providers,
        "transactions": transform_transactions,
    }

    targets = run.keys() if args.entity == "all" else [args.entity]
    for entity in targets:
        run[entity](spark, args.bronze_dir, args.silver_dir)

    spark.stop()
    logger.info("✅ Silver transforms complete.")


if __name__ == "__main__":
    main()
