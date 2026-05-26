"""
MediRisk Intelligence Platform
Tests — PySpark Transform & Feature Engineering
Uses pytest + pyspark local mode for fast unit testing.
Run with: pytest tests/test_transforms.py -v
"""

from __future__ import annotations

import json
from datetime import date, datetime

import pytest
from pyspark.sql import Row, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    BooleanType, DateType, DoubleType, IntegerType,
    StringType, StructField, StructType,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def spark():
    session = (
        SparkSession.builder
        .master("local[2]")
        .appName("MediRisk-Tests")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()


@pytest.fixture
def raw_claims_df(spark):
    schema = StructType([
        StructField("claim_id",               StringType(),  False),
        StructField("patient_id",             StringType(),  False),
        StructField("provider_id",            StringType(),  False),
        StructField("service_date",           StringType(),  True),
        StructField("submitted_date",         StringType(),  True),
        StructField("diagnosis_code",         StringType(),  True),
        StructField("procedure_code",         StringType(),  True),
        StructField("billed_amount",          DoubleType(),  True),
        StructField("allowed_amount",         DoubleType(),  True),
        StructField("paid_amount",            DoubleType(),  True),
        StructField("claim_status",           StringType(),  True),
        StructField("is_duplicate",           BooleanType(), True),
        StructField("is_fraud",               BooleanType(), True),
        StructField("fraud_type",             StringType(),  True),
        StructField("fraud_signals",          StringType(),  True),
        StructField("ingestion_ts",           StringType(),  True),
        StructField("_dq_status",             StringType(),  True),
        StructField("lag_days",               IntegerType(), True),
        StructField("billed_to_allowed_ratio",DoubleType(),  True),
        StructField("service_year",           IntegerType(), True),
    ])
    rows = [
        Row(
            claim_id="c001", patient_id="p001", provider_id="prov001",
            service_date="2024-01-15", submitted_date="2024-01-20",
            diagnosis_code="I10", procedure_code="99213",
            billed_amount=500.0, allowed_amount=400.0, paid_amount=380.0,
            claim_status="APPROVED", is_duplicate=False, is_fraud=False,
            fraud_type=None, fraud_signals='["HIGH_VALUE_CLAIM"]',
            ingestion_ts="2024-01-20T10:00:00",
            _dq_status="PASS", lag_days=5,
            billed_to_allowed_ratio=1.25, service_year=2024,
        ),
        Row(
            claim_id="c002", patient_id="p001", provider_id="prov001",
            service_date="2024-01-20", submitted_date="2024-02-05",
            diagnosis_code="E11.9", procedure_code="99214",
            billed_amount=8000.0, allowed_amount=2000.0, paid_amount=1800.0,
            claim_status="SUBMITTED", is_duplicate=False, is_fraud=True,
            fraud_type="upcoding", fraud_signals='["HIGH_BILLED_RATIO","HIGH_VALUE_CLAIM"]',
            ingestion_ts="2024-02-05T08:00:00",
            _dq_status="PASS", lag_days=16,
            billed_to_allowed_ratio=4.0, service_year=2024,
        ),
        Row(
            claim_id="c003", patient_id="p002", provider_id="prov002",
            service_date="2024-02-01", submitted_date="2024-02-01",
            diagnosis_code="J18.9", procedure_code="71046",
            billed_amount=200.0, allowed_amount=180.0, paid_amount=175.0,
            claim_status="APPROVED", is_duplicate=False, is_fraud=False,
            fraud_type=None, fraud_signals='[]',
            ingestion_ts="2024-02-01T12:00:00",
            _dq_status="PASS", lag_days=0,
            billed_to_allowed_ratio=1.11, service_year=2024,
        ),
        # Quarantine row — should be filtered out
        Row(
            claim_id=None, patient_id="p003", provider_id="prov003",
            service_date="2024-02-10", submitted_date="2024-02-15",
            diagnosis_code="N18.3", procedure_code="99215",
            billed_amount=1000.0, allowed_amount=800.0, paid_amount=780.0,
            claim_status="PENDING", is_duplicate=False, is_fraud=False,
            fraud_type=None, fraud_signals='[]',
            ingestion_ts="2024-02-15T09:00:00",
            _dq_status="QUARANTINE", lag_days=5,
            billed_to_allowed_ratio=1.25, service_year=2024,
        ),
    ]
    return spark.createDataFrame(rows, schema)


@pytest.fixture
def raw_patients_df(spark):
    rows = [
        Row(patient_id="p001", first_name="Alice", last_name="Smith",
            dob="1975-06-15", gender="F", zip_code="10001", state="NY",
            insurance_type="Medicare", insurance_id="MCR-001",
            chronic_conditions=3, created_at="2020-01-01T00:00:00",
            _dq_status="PASS"),
        Row(patient_id="p002", first_name="Bob", last_name="Jones",
            dob="1990-03-22", gender="M", zip_code="90210", state="CA",
            insurance_type="Aetna", insurance_id="ATN-002",
            chronic_conditions=1, created_at="2021-05-10T00:00:00",
            _dq_status="PASS"),
        Row(patient_id="p003", first_name=None, last_name=None,
            dob=None, gender="O", zip_code="00000", state="TX",
            insurance_type="Medicaid", insurance_id=None,
            chronic_conditions=0, created_at="2022-07-01T00:00:00",
            _dq_status="QUARANTINE"),
    ]
    schema = StructType([
        StructField("patient_id",         StringType(),  False),
        StructField("first_name",         StringType(),  True),
        StructField("last_name",          StringType(),  True),
        StructField("dob",                StringType(),  True),
        StructField("gender",             StringType(),  True),
        StructField("zip_code",           StringType(),  True),
        StructField("state",              StringType(),  True),
        StructField("insurance_type",     StringType(),  True),
        StructField("insurance_id",       StringType(),  True),
        StructField("chronic_conditions", IntegerType(), True),
        StructField("created_at",         StringType(),  True),
        StructField("_dq_status",         StringType(),  True),
    ])
    return spark.createDataFrame(rows, schema)


# ─── Bronze DQ Tests ───────────────────────────────────────────────────────────

class TestBronzeDQChecks:

    def test_quarantine_rows_are_tagged(self, raw_claims_df):
        quarantine = raw_claims_df.filter(F.col("_dq_status") == "QUARANTINE")
        assert quarantine.count() == 1

    def test_pass_rows_have_non_null_pk(self, raw_claims_df):
        pass_df = raw_claims_df.filter(F.col("_dq_status") == "PASS")
        null_pks = pass_df.filter(F.col("claim_id").isNull()).count()
        assert null_pks == 0

    def test_pass_rate(self, raw_claims_df):
        total     = raw_claims_df.count()
        passed    = raw_claims_df.filter(F.col("_dq_status") == "PASS").count()
        pass_rate = passed / total
        assert pass_rate == 0.75   # 3 of 4 pass


# ─── Silver Transform Tests ───────────────────────────────────────────────────

class TestSilverTransforms:

    def test_quarantine_rows_are_excluded(self, raw_claims_df):
        silver = raw_claims_df.filter(F.col("_dq_status") == "PASS")
        assert silver.count() == 3

    def test_negative_billed_amount_nullified(self, spark):
        """Negative billed amounts should be set to NULL in Silver."""
        schema = StructType([
            StructField("claim_id",      StringType(), False),
            StructField("billed_amount", DoubleType(), True),
            StructField("_dq_status",    StringType(), True),
        ])
        df = spark.createDataFrame(
            [Row(claim_id="x", billed_amount=-100.0, _dq_status="PASS")],
            schema,
        )
        result = df.withColumn(
            "billed_amount",
            F.when(F.col("billed_amount") < 0, F.lit(None).cast(DoubleType()))
             .otherwise(F.col("billed_amount")),
        )
        assert result.first()["billed_amount"] is None

    def test_allowed_amount_capped_at_billed(self, spark):
        schema = StructType([
            StructField("claim_id",      StringType(), False),
            StructField("billed_amount", DoubleType(), True),
            StructField("allowed_amount",DoubleType(), True),
        ])
        df = spark.createDataFrame(
            [Row(claim_id="x", billed_amount=500.0, allowed_amount=600.0)],
            schema,
        )
        result = df.withColumn(
            "allowed_amount",
            F.when(F.col("allowed_amount") > F.col("billed_amount"), F.col("billed_amount"))
             .otherwise(F.col("allowed_amount")),
        )
        assert result.first()["allowed_amount"] == 500.0

    def test_lag_days_non_negative(self, raw_claims_df):
        df = raw_claims_df.filter(F.col("_dq_status") == "PASS")
        neg_lag = df.filter(F.col("lag_days") < 0).count()
        assert neg_lag == 0

    def test_claim_status_uppercased(self, spark):
        schema = StructType([StructField("claim_status", StringType(), True)])
        df = spark.createDataFrame([Row(claim_status="approved")], schema)
        result = df.withColumn("claim_status", F.upper(F.trim("claim_status")))
        assert result.first()["claim_status"] == "APPROVED"

    def test_deduplication_keeps_latest_submission(self, spark):
        from pyspark.sql import Window

        schema = StructType([
            StructField("claim_id",       StringType(), False),
            StructField("submitted_date", StringType(), True),
            StructField("billed_amount",  DoubleType(), True),
        ])
        rows = [
            Row(claim_id="DUP001", submitted_date="2024-01-10", billed_amount=100.0),
            Row(claim_id="DUP001", submitted_date="2024-01-20", billed_amount=120.0),
            Row(claim_id="DUP001", submitted_date="2024-01-15", billed_amount=110.0),
        ]
        df = spark.createDataFrame(rows, schema)

        win = Window.partitionBy("claim_id").orderBy(F.col("submitted_date").desc())
        deduped = (
            df.withColumn("_rn", F.row_number().over(win))
            .filter(F.col("_rn") == 1)
            .drop("_rn")
        )

        assert deduped.count() == 1
        assert deduped.first()["billed_amount"] == 120.0

    def test_patient_age_band_assignment(self, raw_patients_df):
        df = (
            raw_patients_df
            .filter(F.col("_dq_status") == "PASS")
            .withColumn("dob_date", F.to_date("dob"))
            .withColumn(
                "age",
                F.floor(F.months_between(F.current_date(), F.col("dob_date")) / 12).cast("int"),
            )
            .withColumn(
                "age_band",
                F.when(F.col("age") < 30,  F.lit("18-29"))
                 .when(F.col("age") < 45,  F.lit("30-44"))
                 .when(F.col("age") < 60,  F.lit("45-59"))
                 .when(F.col("age") < 75,  F.lit("60-74"))
                 .otherwise(F.lit("75+")),
            )
        )
        # p001 born 1975 → age ~49 → band 45-59
        alice = df.filter(F.col("patient_id") == "p001").first()
        assert alice["age_band"] == "45-59"

        # p002 born 1990 → age ~34 → band 30-44
        bob = df.filter(F.col("patient_id") == "p002").first()
        assert bob["age_band"] == "30-44"

    def test_high_risk_flag_set_for_3_plus_conditions(self, raw_patients_df):
        df = (
            raw_patients_df
            .filter(F.col("_dq_status") == "PASS")
            .withColumn(
                "high_risk_flag",
                (F.col("chronic_conditions") >= 3).cast("boolean"),
            )
        )
        alice = df.filter(F.col("patient_id") == "p001").first()
        bob   = df.filter(F.col("patient_id") == "p002").first()
        assert alice["high_risk_flag"] is True
        assert bob["high_risk_flag"] is False


# ─── Gold Feature Engineering Tests ──────────────────────────────────────────

class TestGoldFeatures:

    def test_fraud_signals_count_parsed_correctly(self, raw_claims_df):
        df = (
            raw_claims_df
            .filter(F.col("_dq_status") == "PASS")
            .withColumn(
                "fraud_signals_count",
                F.size(F.from_json("fraud_signals", "array<string>")),
            )
        )
        c001 = df.filter(F.col("claim_id") == "c001").first()
        c002 = df.filter(F.col("claim_id") == "c002").first()
        c003 = df.filter(F.col("claim_id") == "c003").first()

        assert c001["fraud_signals_count"] == 1
        assert c002["fraud_signals_count"] == 2
        assert c003["fraud_signals_count"] == 0

    def test_is_high_lag_flag(self, raw_claims_df):
        df = raw_claims_df.filter(F.col("_dq_status") == "PASS")
        df = df.withColumn(
            "is_high_lag", (F.col("lag_days") > 30).cast("int")
        )
        # c002 has lag_days=16, not > 30
        c002 = df.filter(F.col("claim_id") == "c002").first()
        assert c002["is_high_lag"] == 0

    def test_billed_vs_provider_avg(self, spark):
        schema = StructType([
            StructField("claim_id",            StringType(), False),
            StructField("billed_amount",       DoubleType(), True),
            StructField("prov_90d_avg_billed", DoubleType(), True),
        ])
        df = spark.createDataFrame(
            [Row(claim_id="x", billed_amount=1000.0, prov_90d_avg_billed=500.0)],
            schema,
        )
        result = df.withColumn(
            "billed_vs_provider_avg",
            F.round(F.col("billed_amount") / (F.col("prov_90d_avg_billed") + 1), 4),
        )
        assert abs(result.first()["billed_vs_provider_avg"] - 1.996) < 0.001

    def test_composite_risk_score_bounded_0_100(self, raw_claims_df):
        """Risk score should never exceed 100."""
        df = (
            raw_claims_df
            .filter(F.col("_dq_status") == "PASS")
            .withColumn(
                "composite_risk_score",
                F.least(
                    F.lit(100.0),
                    F.col("billed_to_allowed_ratio") * 30.0 + F.lit(10.0),
                ),
            )
        )
        max_score = df.agg(F.max("composite_risk_score")).collect()[0][0]
        assert max_score <= 100.0

    def test_risk_tier_assignment(self, spark):
        schema = StructType([StructField("score", DoubleType(), True)])
        df = spark.createDataFrame([
            Row(score=80.0),
            Row(score=55.0),
            Row(score=20.0),
        ], schema)

        df = df.withColumn(
            "risk_tier",
            F.when(F.col("score") >= 70, F.lit("HIGH"))
             .when(F.col("score") >= 40, F.lit("MEDIUM"))
             .otherwise(F.lit("LOW")),
        )
        tiers = {row["score"]: row["risk_tier"] for row in df.collect()}
        assert tiers[80.0] == "HIGH"
        assert tiers[55.0] == "MEDIUM"
        assert tiers[20.0] == "LOW"

    def test_patient_aggregation_totals(self, raw_claims_df):
        df = raw_claims_df.filter(F.col("_dq_status") == "PASS")

        agg = (
            df
            .groupBy("patient_id")
            .agg(
                F.count("claim_id").alias("total_claims"),
                F.sum("billed_amount").alias("lifetime_billed"),
                F.sum(F.col("is_fraud").cast("int")).alias("fraud_count"),
            )
        )

        # p001 has c001 + c002
        p001 = agg.filter(F.col("patient_id") == "p001").first()
        assert p001["total_claims"] == 2
        assert abs(p001["lifetime_billed"] - 8500.0) < 0.01
        assert p001["fraud_count"] == 1

        # p002 has c003 only
        p002 = agg.filter(F.col("patient_id") == "p002").first()
        assert p002["total_claims"] == 1
        assert p002["fraud_count"] == 0

    def test_fraud_rate_calculation(self, spark):
        schema = StructType([
            StructField("patient_id",   StringType(), False),
            StructField("total_claims", IntegerType(), True),
            StructField("fraud_count",  IntegerType(), True),
        ])
        df = spark.createDataFrame(
            [Row(patient_id="p001", total_claims=10, fraud_count=3)],
            schema,
        )
        result = df.withColumn(
            "fraud_rate",
            F.round(F.col("fraud_count") / F.col("total_claims"), 4),
        )
        assert result.first()["fraud_rate"] == 0.3


# ─── Window Function Tests ─────────────────────────────────────────────────────

class TestWindowFunctions:

    def test_provider_rolling_count(self, spark):
        from pyspark.sql import Window

        schema = StructType([
            StructField("claim_id",    StringType(), False),
            StructField("provider_id", StringType(), False),
            StructField("service_date",StringType(), True),
            StructField("is_fraud",    BooleanType(),True),
        ])
        rows = [
            Row(claim_id="a", provider_id="P1", service_date="2024-01-01", is_fraud=False),
            Row(claim_id="b", provider_id="P1", service_date="2024-01-15", is_fraud=True),
            Row(claim_id="c", provider_id="P1", service_date="2024-02-10", is_fraud=False),
            Row(claim_id="d", provider_id="P2", service_date="2024-01-01", is_fraud=False),
        ]
        df = spark.createDataFrame(rows, schema).withColumn(
            "service_date", F.to_date("service_date")
        )

        win = (
            Window.partitionBy("provider_id")
            .orderBy(F.col("service_date").cast("long"))
            .rangeBetween(-90 * 86_400, 0)
        )
        df = df.withColumn("prov_90d_count", F.count("claim_id").over(win))

        p1_claims = {
            row["claim_id"]: row["prov_90d_count"]
            for row in df.filter(F.col("provider_id") == "P1").collect()
        }

        # All 3 P1 claims fall within 90 days of each other
        assert p1_claims["c"] == 3

        # P2 has only 1 claim
        p2 = df.filter(F.col("provider_id") == "P2").first()
        assert p2["prov_90d_count"] == 1
