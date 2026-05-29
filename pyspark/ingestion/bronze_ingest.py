"""
MediRisk Intelligence Platform
Bronze Layer — Raw Ingestion
Reads raw CSV/Parquet sources and writes to Delta Lake Bronze tables
with schema enforcement, source metadata, and data quality tagging.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    BooleanType, DateType, DoubleType, IntegerType,
    StringType, StructField, StructType, TimestampType,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ─── Schema Definitions ───────────────────────────────────────────────────────

CLAIMS_SCHEMA = StructType([
    StructField("claim_id",        StringType(),  nullable=False),
    StructField("patient_id",      StringType(),  nullable=False),
    StructField("provider_id",     StringType(),  nullable=False),
    StructField("service_date",    StringType(),  nullable=True),
    StructField("submitted_date",  StringType(),  nullable=True),
    StructField("diagnosis_code",  StringType(),  nullable=True),
    StructField("procedure_code",  StringType(),  nullable=True),
    StructField("billed_amount",   DoubleType(),  nullable=True),
    StructField("allowed_amount",  DoubleType(),  nullable=True),
    StructField("paid_amount",     DoubleType(),  nullable=True),
    StructField("claim_status",    StringType(),  nullable=True),
    StructField("is_duplicate",    BooleanType(), nullable=True),
    StructField("is_fraud",        BooleanType(), nullable=True),
    StructField("fraud_type",      StringType(),  nullable=True),
    StructField("fraud_signals",   StringType(),  nullable=True),
    StructField("ingestion_ts",    StringType(),  nullable=True),
])

PATIENTS_SCHEMA = StructType([
    StructField("patient_id",          StringType(),  nullable=False),
    StructField("first_name",          StringType(),  nullable=True),
    StructField("last_name",           StringType(),  nullable=True),
    StructField("dob",                 StringType(),  nullable=True),
    StructField("gender",              StringType(),  nullable=True),
    StructField("zip_code",            StringType(),  nullable=True),
    StructField("state",               StringType(),  nullable=True),
    StructField("insurance_type",      StringType(),  nullable=True),
    StructField("insurance_id",        StringType(),  nullable=True),
    StructField("chronic_conditions",  IntegerType(), nullable=True),
    StructField("created_at",          StringType(),  nullable=True),
])

PROVIDERS_SCHEMA = StructType([
    StructField("provider_id",   StringType(),  nullable=False),
    StructField("npi",           StringType(),  nullable=True),
    StructField("name",          StringType(),  nullable=True),
    StructField("specialty",     StringType(),  nullable=True),
    StructField("state",         StringType(),  nullable=True),
    StructField("zip_code",      StringType(),  nullable=True),
    StructField("is_flagged",    BooleanType(), nullable=True),
    StructField("license_valid", BooleanType(), nullable=True),
    StructField("joined_date",   StringType(),  nullable=True),
])

TRANSACTIONS_SCHEMA = StructType([
    StructField("transaction_id",   StringType(), nullable=False),
    StructField("patient_id",       StringType(), nullable=False),
    StructField("transaction_type", StringType(), nullable=True),
    StructField("amount",           DoubleType(), nullable=True),
    StructField("transaction_date", StringType(), nullable=True),
    StructField("channel",          StringType(), nullable=True),
    StructField("is_anomalous",     BooleanType(),nullable=True),
    StructField("ingestion_ts",     StringType(), nullable=True),
])


# ─── Helpers ──────────────────────────────────────────────────────────────────

def get_spark(app_name: str = "MediRisk-Bronze") -> SparkSession:
    return (
        SparkSession.builder
        .appName(app_name)
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        .config("spark.sql.shuffle.partitions", "200")
        .config("spark.databricks.delta.schema.autoMerge.enabled", "true")
        .getOrCreate()
    )


def add_bronze_metadata(df: DataFrame, source_path: str, source_name: str) -> DataFrame:
    """Attach audit columns to every Bronze record."""
    return df.withColumn("_source_path",   F.lit(source_path)) \
             .withColumn("_source_name",   F.lit(source_name)) \
             .withColumn("_bronze_ts",     F.current_timestamp()) \
             .withColumn("_batch_id",      F.lit(datetime.utcnow().strftime("%Y%m%d%H%M%S"))) \
             .withColumn("_record_hash",   F.md5(F.to_json(F.struct("*"))))


def run_dq_checks(df: DataFrame, pk_col: str, not_null_cols: list[str]) -> DataFrame:
    """
    Tag each row with a DQ status flag.
    Rows with PK nulls or critical-column nulls are marked QUARANTINE.
    """
    condition = F.col(pk_col).isNotNull()
    for col in not_null_cols:
        condition = condition & F.col(col).isNotNull()

    return df.withColumn(
        "_dq_status",
        F.when(condition, F.lit("PASS")).otherwise(F.lit("QUARANTINE")),
    )


def write_bronze(
    df: DataFrame,
    delta_path: str,
    partition_cols: Optional[list[str]] = None,
    mode: str = "append",
) -> None:
    writer = (
        df.write
        .format("delta")
        .mode(mode)
        .option("mergeSchema", "true")
    )
    if partition_cols:
        writer = writer.partitionBy(*partition_cols)
    writer.save(delta_path)
    logger.info("Wrote %d rows to %s", df.count(), delta_path)


# ─── Ingestion Functions ───────────────────────────────────────────────────────

def ingest_claims(spark: SparkSession, source_path: str, bronze_path: str) -> None:
    logger.info("Ingesting claims from %s", source_path)

    df = (
        spark.read
        .schema(CLAIMS_SCHEMA)
        .parquet(source_path)
    )

    df = add_bronze_metadata(df, source_path, "claims")
    df = run_dq_checks(df, pk_col="claim_id", not_null_cols=["patient_id", "provider_id"])

    # Partition by service year/month for efficient downstream reads
    df = df.withColumn(
        "_service_year_month",
        F.date_format(F.to_date("service_date"), "yyyy-MM"),
    )

    write_bronze(df, f"{bronze_path}/claims", partition_cols=["_service_year_month"])
    _log_stats("claims", df)


def ingest_patients(spark: SparkSession, source_path: str, bronze_path: str) -> None:
    logger.info("Ingesting patients from %s", source_path)

    df = (
        spark.read
        .schema(PATIENTS_SCHEMA)
        .option("header", "true")
        .csv(source_path)
    )

    df = add_bronze_metadata(df, source_path, "patients")
    df = run_dq_checks(df, pk_col="patient_id", not_null_cols=["dob", "insurance_type"])

    write_bronze(df, f"{bronze_path}/patients", partition_cols=["state"])
    _log_stats("patients", df)


def ingest_providers(spark: SparkSession, source_path: str, bronze_path: str) -> None:
    logger.info("Ingesting providers from %s", source_path)

    df = (
        spark.read
        .schema(PROVIDERS_SCHEMA)
        .option("header", "true")
        .csv(source_path)
    )

    df = add_bronze_metadata(df, source_path, "providers")
    df = run_dq_checks(df, pk_col="provider_id", not_null_cols=["npi", "specialty"])

    write_bronze(df, f"{bronze_path}/providers", partition_cols=["state"])
    _log_stats("providers", df)


def ingest_transactions(spark: SparkSession, source_path: str, bronze_path: str) -> None:
    logger.info("Ingesting transactions from %s", source_path)

    df = (
        spark.read
        .schema(TRANSACTIONS_SCHEMA)
        .parquet(source_path)
    )

    df = add_bronze_metadata(df, source_path, "transactions")
    df = run_dq_checks(df, pk_col="transaction_id", not_null_cols=["patient_id", "amount"])

    df = df.withColumn(
        "_tx_year_month",
        F.date_format(F.to_date("transaction_date"), "yyyy-MM"),
    )

    write_bronze(df, f"{bronze_path}/transactions", partition_cols=["_tx_year_month"])
    _log_stats("transactions", df)


def _log_stats(name: str, df: DataFrame) -> None:
    total       = df.count() 
    quarantined = df.filter(F.col("_dq_status") == "QUARANTINE").count()
    logger.info(
        "[%s] Total: %d  |  Quarantined: %d  |  Pass rate: %.2f%%",
        name, total, quarantined,
        (total - quarantined) / total * 100 if total else 0,
    )


# ─── Entry Point ──────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="MediRisk Bronze Ingestion")
    parser.add_argument("--source-dir", required=True,  help="Path to raw data directory")
    parser.add_argument("--bronze-dir", required=True,  help="Delta Lake Bronze output path")
    parser.add_argument("--entity",     default="all",  help="Entity to ingest: claims|patients|providers|transactions|all")
    args = parser.parse_args()

    spark = get_spark()

    if args.entity in ("claims", "all"):
        ingest_claims(spark, f"{args.source_dir}/claims.parquet", args.bronze_dir)
    if args.entity in ("patients", "all"):
        ingest_patients(spark, f"{args.source_dir}/patients.csv", args.bronze_dir)
    if args.entity in ("providers", "all"):
        ingest_providers(spark, f"{args.source_dir}/providers.csv", args.bronze_dir)
    if args.entity in ("transactions", "all"):
        ingest_transactions(spark, f"{args.source_dir}/transactions.parquet", args.bronze_dir)

    spark.stop()
    logger.info("✅ Bronze ingestion complete.")


if __name__ == "__main__":
    main()
