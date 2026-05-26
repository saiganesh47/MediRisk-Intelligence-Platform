"""
MediRisk Intelligence Platform
Snowpark Python — Risk Score Engine Stored Procedure
Advanced business rule engine that runs inside Snowflake using Snowpark,
computes composite risk scores, and writes results to GOLD schema.

Deploy with:
    snowsql -c medirisk -f risk_score_proc.py
or via Snowflake Python API / SnowCLI.
"""

from __future__ import annotations

import json
from datetime import datetime

from snowflake.snowpark import Session
from snowflake.snowpark import functions as F
from snowflake.snowpark.types import (
    DoubleType, IntegerType, StringType, StructField, StructType,
    TimestampType,
)


# ─── Risk Weights (externalised so they can be tuned without redeploying) ─────

DEFAULT_WEIGHTS = {
    "fraud_probability_weight":   0.40,
    "chronic_condition_weight":   0.05,  # per condition
    "provider_risk_weight":       0.20,
    "anomalous_tx_weight":        0.15,
    "high_risk_claim_weight":     0.02,  # per high-risk claim
    "provider_diversity_penalty": 0.10,  # if distinct_providers > 5
    "late_submission_penalty":    0.08,  # if avg_lag > 30 days
}


# ─── Helper: load weights from a Snowflake config table if present ─────────────

def _load_weights(session: Session) -> dict:
    try:
        rows = (
            session.table("MEDIRISK.GOLD.RISK_WEIGHTS_CONFIG")
            .select("WEIGHT_KEY", "WEIGHT_VALUE")
            .collect()
        )
        if rows:
            loaded = {r["WEIGHT_KEY"]: float(r["WEIGHT_VALUE"]) for r in rows}
            return {**DEFAULT_WEIGHTS, **loaded}
    except Exception:
        pass
    return DEFAULT_WEIGHTS


# ─── Core Scoring Function ────────────────────────────────────────────────────

def compute_patient_risk_scores(session: Session, weights: dict | None = None) -> str:
    """
    Joins GOLD.PATIENT_RISK_SUMMARY with SILVER fraud data to compute
    a weighted composite risk score per patient. Writes results to
    GOLD.PATIENT_RISK_SCORES_FINAL.
    """
    if weights is None:
        weights = _load_weights(session)

    # Read source tables
    patient_summary = session.table("MEDIRISK.GOLD.PATIENT_RISK_SUMMARY")
    fraud_scores    = session.table("MEDIRISK.SILVER.FRAUD_SCORES")
    providers       = session.table("MEDIRISK.RAW.PROVIDERS_RAW")

    # Patient-level fraud aggregates
    fraud_agg = (
        fraud_scores
        .group_by("PATIENT_ID")
        .agg(
            F.avg("FRAUD_PROBABILITY").alias("avg_fraud_prob"),
            F.sum("FRAUD_FLAG").alias("total_fraud_flags"),
            F.count("CLAIM_ID").alias("scored_claim_count"),
            F.count_distinct("PROVIDER_ID").alias("distinct_providers_scored"),
        )
    )

    # Provider risk lookup (max risk score across providers used by patient)
    provider_patient = (
        fraud_scores
        .join(providers.select("PROVIDER_ID", "PROVIDER_RISK_SCORE"), "PROVIDER_ID", "left")
        .group_by("PATIENT_ID")
        .agg(
            F.avg("PROVIDER_RISK_SCORE").alias("avg_provider_risk_score"),
            F.max("PROVIDER_RISK_SCORE").alias("max_provider_risk_score"),
        )
    )

    # Join everything
    scored = (
        patient_summary
        .join(fraud_agg,       "PATIENT_ID", "left")
        .join(provider_patient, "PATIENT_ID", "left")
        .na.fill(0.0, subset=[
            "avg_fraud_prob", "total_fraud_flags", "scored_claim_count",
            "distinct_providers_scored", "avg_provider_risk_score",
        ])
    )

    # Compute composite score in Snowpark
    w = weights

    scored = scored.with_column(
        "RAW_RISK_SCORE",
        (
            F.col("avg_fraud_prob")               * w["fraud_probability_weight"]   * 100 +
            F.col("CHRONIC_CONDITIONS")            * w["chronic_condition_weight"]   * 100 +
            F.col("avg_provider_risk_score") / 100 * w["provider_risk_weight"]       * 100 +
            F.col("HIGH_RISK_CLAIM_COUNT")         * w["high_risk_claim_weight"]     * 100 +
            F.when(
                F.col("DISTINCT_PROVIDERS_SCORED") > 5,
                F.lit(w["provider_diversity_penalty"] * 100),
            ).otherwise(F.lit(0.0)) +
            F.when(
                F.col("AVG_SUBMISSION_LAG") > 30,
                F.lit(w["late_submission_penalty"] * 100),
            ).otherwise(F.lit(0.0))
        ),
    )

    scored = scored.with_column(
        "COMPOSITE_RISK_SCORE",
        F.least(F.lit(100.0), F.round(F.col("RAW_RISK_SCORE"), 2)),
    )

    scored = scored.with_column(
        "RISK_TIER",
        F.when(F.col("COMPOSITE_RISK_SCORE") >= 70, F.lit("HIGH"))
         .when(F.col("COMPOSITE_RISK_SCORE") >= 40, F.lit("MEDIUM"))
         .otherwise(F.lit("LOW")),
    )

    scored = scored.with_column(
        "SCORE_COMPUTED_AT", F.current_timestamp()
    ).with_column(
        "WEIGHTS_SNAPSHOT", F.lit(json.dumps(weights))
    )

    # Select final output columns
    output_cols = [
        "PATIENT_ID", "FIRST_NAME", "LAST_NAME", "AGE", "AGE_BAND",
        "STATE", "INSURANCE_TYPE", "CHRONIC_CONDITIONS", "HIGH_RISK_FLAG",
        "TOTAL_CLAIMS", "LIFETIME_BILLED", "AVG_FRAUD_PROBABILITY",
        "FRAUD_CLAIM_COUNT", "FRAUD_CLAIM_RATE", "avg_fraud_prob",
        "DISTINCT_PROVIDERS_SCORED", "avg_provider_risk_score",
        "max_provider_risk_score", "COMPOSITE_RISK_SCORE",
        "RISK_TIER", "SCORE_COMPUTED_AT", "WEIGHTS_SNAPSHOT",
    ]

    output = scored.select(*output_cols)

    # Write to GOLD
    (
        output.write
        .mode("overwrite")
        .save_as_table("MEDIRISK.GOLD.PATIENT_RISK_SCORES_FINAL")
    )

    row_count = output.count()
    return f"SUCCESS: Computed risk scores for {row_count} patients. Weights: {json.dumps(weights)}"


# ─── Anomaly Detection: Provider Billing Outliers ─────────────────────────────

def flag_provider_billing_outliers(session: Session, z_threshold: float = 3.0) -> str:
    """
    Flags providers whose average billed amount is more than z_threshold
    standard deviations above the specialty mean. Writes to GOLD.PROVIDER_OUTLIERS.
    """
    fraud_scores = session.table("MEDIRISK.SILVER.FRAUD_SCORES")
    providers    = session.table("MEDIRISK.RAW.PROVIDERS_RAW")

    # Provider billing stats
    prov_billing = (
        fraud_scores
        .group_by("PROVIDER_ID")
        .agg(
            F.avg("BILLED_AMOUNT").alias("avg_billed"),
            F.count("CLAIM_ID").alias("total_claims"),
        )
        .join(providers.select("PROVIDER_ID", "SPECIALTY", "IS_FLAGGED"), "PROVIDER_ID", "left")
    )

    # Specialty-level stats
    specialty_stats = (
        prov_billing
        .group_by("SPECIALTY")
        .agg(
            F.avg("avg_billed").alias("specialty_avg"),
            F.stddev("avg_billed").alias("specialty_std"),
        )
    )

    # Z-score per provider within specialty
    outliers = (
        prov_billing
        .join(specialty_stats, "SPECIALTY", "left")
        .with_column(
            "Z_SCORE",
            F.when(
                F.col("specialty_std") > 0,
                (F.col("avg_billed") - F.col("specialty_avg")) / F.col("specialty_std"),
            ).otherwise(F.lit(0.0)),
        )
        .filter(F.col("Z_SCORE") > z_threshold)
        .with_column("FLAGGED_AT", F.current_timestamp())
        .sort(F.col("Z_SCORE").desc())
    )

    outliers.write.mode("overwrite").save_as_table("MEDIRISK.GOLD.PROVIDER_OUTLIERS")

    count = outliers.count()
    return f"SUCCESS: Flagged {count} provider billing outliers (z > {z_threshold})."


# ─── Stored Procedure Entrypoints ─────────────────────────────────────────────
# These are the functions registered as Snowflake Stored Procedures.

def run_risk_scoring(session: Session) -> str:
    """Registered as: CALL MEDIRISK.GOLD.RUN_RISK_SCORING()"""
    try:
        weights = _load_weights(session)
        return compute_patient_risk_scores(session, weights)
    except Exception as e:
        return f"ERROR: {str(e)}"


def run_outlier_detection(session: Session, z_threshold: float = 3.0) -> str:
    """Registered as: CALL MEDIRISK.GOLD.RUN_OUTLIER_DETECTION(3.0)"""
    try:
        return flag_provider_billing_outliers(session, z_threshold)
    except Exception as e:
        return f"ERROR: {str(e)}"


# ─── Registration SQL (run once to register procedures in Snowflake) ──────────
REGISTRATION_SQL = """
-- Register risk scoring procedure
CREATE OR REPLACE PROCEDURE MEDIRISK.GOLD.RUN_RISK_SCORING()
RETURNS STRING
LANGUAGE PYTHON
RUNTIME_VERSION = '3.11'
PACKAGES        = ('snowflake-snowpark-python')
HANDLER         = 'risk_score_proc.run_risk_scoring';

-- Register outlier detection procedure
CREATE OR REPLACE PROCEDURE MEDIRISK.GOLD.RUN_OUTLIER_DETECTION(Z_THRESHOLD FLOAT)
RETURNS STRING
LANGUAGE PYTHON
RUNTIME_VERSION = '3.11'
PACKAGES        = ('snowflake-snowpark-python')
HANDLER         = 'risk_score_proc.run_outlier_detection';

-- Schedule via Task
CREATE OR REPLACE TASK MEDIRISK.GOLD.DAILY_RISK_SCORING_TASK
    WAREHOUSE = MEDIRISK_TRANSFORM_WH
    SCHEDULE  = 'USING CRON 0 2 * * * UTC'
    COMMENT   = 'Runs nightly at 02:00 UTC'
AS
    CALL MEDIRISK.GOLD.RUN_RISK_SCORING();

ALTER TASK MEDIRISK.GOLD.DAILY_RISK_SCORING_TASK RESUME;
"""
