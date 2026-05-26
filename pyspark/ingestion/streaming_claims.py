"""
MediRisk Intelligence Platform
Streaming Bronze Layer — Structured Streaming for Real-Time Claims
Consumes claims from Kafka or a landing-zone file stream,
applies schema validation, DQ tagging, and writes to Delta Bronze.
"""

from __future__ import annotations

import logging

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    BooleanType, DoubleType, StringType, StructField, StructType,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ─── Schema ───────────────────────────────────────────────────────────────────

CLAIM_EVENT_SCHEMA = StructType([
    StructField("claim_id",       StringType(),  nullable=False),
    StructField("patient_id",     StringType(),  nullable=False),
    StructField("provider_id",    StringType(),  nullable=False),
    StructField("service_date",   StringType(),  nullable=True),
    StructField("diagnosis_code", StringType(),  nullable=True),
    StructField("procedure_code", StringType(),  nullable=True),
    StructField("billed_amount",  DoubleType(),  nullable=True),
    StructField("claim_status",   StringType(),  nullable=True),
    StructField("fraud_signals",  StringType(),  nullable=True),
    StructField("is_fraud",       BooleanType(), nullable=True),
    StructField("ingestion_ts",   StringType(),  nullable=True),
])


# ─── Spark Session ────────────────────────────────────────────────────────────

def get_spark(app_name: str = "MediRisk-Streaming-Bronze") -> SparkSession:
    return (
        SparkSession.builder
        .appName(app_name)
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        # Streaming-specific tuning
        .config("spark.streaming.stopGracefullyOnShutdown", "true")
        .config("spark.sql.streaming.schemaInference",      "false")
        .getOrCreate()
    )


# ─── Source: Kafka ────────────────────────────────────────────────────────────

def read_from_kafka(spark: SparkSession, bootstrap_servers: str, topic: str) -> DataFrame:
    """
    Read raw claim events from Kafka.
    Each Kafka value is expected to be a JSON-encoded claim record.
    """
    raw = (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", bootstrap_servers)
        .option("subscribe",               topic)
        .option("startingOffsets",         "latest")
        .option("maxOffsetsPerTrigger",    10_000)
        .option("failOnDataLoss",          "false")
        .load()
    )

    # Decode binary Kafka value → parse JSON → apply schema
    parsed = (
        raw
        .select(
            F.col("offset").alias("_kafka_offset"),
            F.col("partition").alias("_kafka_partition"),
            F.col("timestamp").alias("_kafka_ts"),
            F.from_json(
                F.col("value").cast("string"),
                CLAIM_EVENT_SCHEMA,
            ).alias("data"),
        )
        .select("_kafka_offset", "_kafka_partition", "_kafka_ts", "data.*")
    )

    return parsed


# ─── Source: File Stream (landing zone) ───────────────────────────────────────

def read_from_file_stream(spark: SparkSession, landing_path: str) -> DataFrame:
    """
    Read new Parquet files dropped into a landing zone directory.
    Useful for batch-near-real-time ingestion without Kafka.
    """
    return (
        spark.readStream
        .schema(CLAIM_EVENT_SCHEMA)
        .option("maxFilesPerTrigger", 5)
        .parquet(landing_path)
        .withColumn("_kafka_offset",    F.lit(None).cast("long"))
        .withColumn("_kafka_partition", F.lit(None).cast("int"))
        .withColumn("_kafka_ts",        F.current_timestamp())
    )


# ─── Transformations ──────────────────────────────────────────────────────────

def enrich_stream(df: DataFrame) -> DataFrame:
    """Apply DQ checks, hash, and bronze metadata to streaming records."""
    return (
        df
        # DQ flag
        .withColumn(
            "_dq_status",
            F.when(
                F.col("claim_id").isNotNull() & F.col("patient_id").isNotNull(),
                F.lit("PASS"),
            ).otherwise(F.lit("QUARANTINE")),
        )
        # Deterministic record hash for dedup
        .withColumn(
            "_record_hash",
            F.md5(F.concat_ws("|", "claim_id", "patient_id", "billed_amount", "service_date")),
        )
        # Partition column
        .withColumn(
            "_service_year_month",
            F.date_format(F.to_date("service_date"), "yyyy-MM"),
        )
        .withColumn("_bronze_ts", F.current_timestamp())
        .withColumn("_source_name", F.lit("streaming_claims"))
    )


# ─── Sinks ────────────────────────────────────────────────────────────────────

def write_to_delta(df: DataFrame, delta_path: str, checkpoint_path: str):
    """
    Write enriched streaming records to Delta Bronze using foreachBatch.
    Applies MERGE to avoid duplicates on claim_id.
    """

    def upsert_to_delta(batch_df: DataFrame, batch_id: int):
        from delta.tables import DeltaTable

        if batch_df.isEmpty():
            return

        batch_df = batch_df.dropDuplicates(["claim_id"])

        if DeltaTable.isDeltaTable(batch_df.sparkSession, delta_path):
            target = DeltaTable.forPath(batch_df.sparkSession, delta_path)
            (
                target.alias("t")
                .merge(batch_df.alias("s"), "t.claim_id = s.claim_id")
                .whenMatchedUpdateAll()
                .whenNotMatchedInsertAll()
                .execute()
            )
        else:
            (
                batch_df.write
                .format("delta")
                .mode("append")
                .partitionBy("_service_year_month")
                .save(delta_path)
            )

        logger.info("Batch %d: upserted %d records to %s", batch_id, batch_df.count(), delta_path)

    return (
        df.writeStream
        .foreachBatch(upsert_to_delta)
        .option("checkpointLocation", checkpoint_path)
        .trigger(processingTime="30 seconds")
        .start()
    )


def write_fraud_alerts(df: DataFrame, alert_path: str, checkpoint_path: str):
    """
    Side-channel: write HIGH-confidence fraud signals to a separate path
    so downstream alerting systems can consume immediately.
    """
    fraud_df = df.filter(
        (F.col("is_fraud") == True) |
        F.col("fraud_signals").contains("HIGH_VALUE_CLAIM")
    )

    return (
        fraud_df.writeStream
        .format("delta")
        .outputMode("append")
        .option("checkpointLocation", checkpoint_path)
        .option("path", alert_path)
        .trigger(processingTime="30 seconds")
        .start()
    )


# ─── Entry Point ──────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="MediRisk Streaming Bronze")
    parser.add_argument("--source",        choices=["kafka", "files"], default="files")
    parser.add_argument("--bootstrap",     default="localhost:9092", help="Kafka bootstrap servers")
    parser.add_argument("--topic",         default="medirisk.claims")
    parser.add_argument("--landing-path",  default="s3a://medirisk-landing/claims/")
    parser.add_argument("--bronze-path",   default="s3a://medirisk-delta/bronze/claims/")
    parser.add_argument("--alert-path",    default="s3a://medirisk-delta/bronze/fraud_alerts/")
    parser.add_argument("--checkpoint",    default="s3a://medirisk-checkpoints/bronze/claims/")
    parser.add_argument("--alert-checkpoint", default="s3a://medirisk-checkpoints/bronze/alerts/")
    args = parser.parse_args()

    spark = get_spark()

    # Read
    if args.source == "kafka":
        raw_stream = read_from_kafka(spark, args.bootstrap, args.topic)
    else:
        raw_stream = read_from_file_stream(spark, args.landing_path)

    # Transform
    enriched = enrich_stream(raw_stream)

    # Write — main sink + fraud alert side-channel
    main_query  = write_to_delta(enriched, args.bronze_path, args.checkpoint)
    alert_query = write_fraud_alerts(enriched, args.alert_path, args.alert_checkpoint)

    logger.info("🚀 Streaming queries started. Awaiting termination...")
    spark.streams.awaitAnyTermination()


if __name__ == "__main__":
    main()
