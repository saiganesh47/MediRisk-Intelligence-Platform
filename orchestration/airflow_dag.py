"""
MediRisk Intelligence Platform
Airflow DAG — End-to-End Pipeline Orchestration
Runs daily at 01:00 UTC:
  1. Generate / fetch raw data
  2. PySpark Bronze ingestion
  3. PySpark Silver transformation
  4. PySpark Gold feature engineering
  5. PySpark ML fraud detection training (weekly) or batch scoring (daily)
  6. Snowflake: refresh Snowpipe / trigger CDC tasks
  7. Snowpark: run risk scoring procedure
  8. Notify on failure
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.models import Variable
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.providers.snowflake.operators.snowflake import SnowflakeOperator
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from airflow.utils.dates import days_ago
from airflow.utils.trigger_rule import TriggerRule


# ─── Config ───────────────────────────────────────────────────────────────────

SNOWFLAKE_CONN_ID = "snowflake_medirisk"
SPARK_CONN_ID     = "spark_medirisk"

BASE_DELTA = Variable.get("medirisk_delta_path",   default_var="s3a://medirisk-delta")
BASE_LAND  = Variable.get("medirisk_landing_path", default_var="s3a://medirisk-landing")
MODEL_DIR  = Variable.get("medirisk_model_dir",    default_var="s3a://medirisk-models")

PYSPARK_PKGS = (
    "io.delta:delta-core_2.12:2.4.0,"
    "org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.0"
)

DEFAULT_ARGS = {
    "owner":            "data-engineering",
    "depends_on_past":  False,
    "email_on_failure": True,
    "email":            ["alerts@medirisk.internal"],
    "retries":          2,
    "retry_delay":      timedelta(minutes=5),
    "execution_timeout": timedelta(hours=2),
}


# ─── Branch: train weekly or score daily ──────────────────────────────────────

def branch_train_or_score(**context) -> str:
    """Retrain model on Sundays; score on all other days."""
    dow = context["logical_date"].weekday()
    return "train_fraud_model" if dow == 6 else "batch_score_claims"


# ─── DAG ──────────────────────────────────────────────────────────────────────

with DAG(
    dag_id="medirisk_pipeline",
    description="MediRisk end-to-end fraud & risk pipeline",
    default_args=DEFAULT_ARGS,
    schedule_interval="0 1 * * *",   # 01:00 UTC daily
    start_date=days_ago(1),
    catchup=False,
    max_active_runs=1,
    tags=["medirisk", "pyspark", "snowflake", "ml"],
    doc_md="""
## MediRisk Intelligence Platform — Daily Pipeline

Runs the full Bronze → Silver → Gold → ML → Snowflake → Snowpark chain.

### Steps
1. **bronze_ingest** — PySpark ingest raw files to Delta Bronze
2. **silver_transform** — Clean & standardise to Silver
3. **gold_features** — Feature engineering & patient risk scoring
4. **branch** — Weekly model retrain OR daily batch scoring
5. **snowpipe_refresh** — Trigger Snowflake Snowpipe loads
6. **cdc_tasks** — Resume CDC Stream+Task pipeline
7. **snowpark_risk** — Run Snowpark risk scoring procedure
8. **verify_dynamic_tables** — Check Dynamic Table freshness
""",
) as dag:

    # ── Start ──────────────────────────────────────────────────────────────────
    start = EmptyOperator(task_id="start")

    # ── Bronze ─────────────────────────────────────────────────────────────────
    bronze_ingest = SparkSubmitOperator(
        task_id="bronze_ingest",
        conn_id=SPARK_CONN_ID,
        application="pyspark/ingestion/bronze_ingest.py",
        application_args=[
            "--source-dir", f"{BASE_LAND}/raw",
            "--bronze-dir", f"{BASE_DELTA}/bronze",
            "--entity",     "all",
        ],
        packages=PYSPARK_PKGS,
        conf={
            "spark.sql.shuffle.partitions": "200",
            "spark.dynamicAllocation.enabled": "true",
        },
        name="medirisk-bronze-{{ ds_nodash }}",
    )

    # ── Silver ─────────────────────────────────────────────────────────────────
    silver_transform = SparkSubmitOperator(
        task_id="silver_transform",
        conn_id=SPARK_CONN_ID,
        application="pyspark/transforms/silver_clean.py",
        application_args=[
            "--bronze-dir", f"{BASE_DELTA}/bronze",
            "--silver-dir", f"{BASE_DELTA}/silver",
            "--entity",     "all",
        ],
        packages=PYSPARK_PKGS,
        name="medirisk-silver-{{ ds_nodash }}",
    )

    # ── Gold Feature Engineering ────────────────────────────────────────────────
    gold_features = SparkSubmitOperator(
        task_id="gold_features",
        conn_id=SPARK_CONN_ID,
        application="pyspark/transforms/gold_features.py",
        application_args=[
            "--silver-dir", f"{BASE_DELTA}/silver",
            "--gold-dir",   f"{BASE_DELTA}/gold",
        ],
        packages=PYSPARK_PKGS,
        conf={"spark.sql.shuffle.partitions": "400"},
        name="medirisk-gold-{{ ds_nodash }}",
    )

    # ── ML Branch ─────────────────────────────────────────────────────────────
    branch_ml = BranchPythonOperator(
        task_id="branch_train_or_score",
        python_callable=branch_train_or_score,
    )

    train_fraud_model = SparkSubmitOperator(
        task_id="train_fraud_model",
        conn_id=SPARK_CONN_ID,
        application="pyspark/ml/fraud_detection.py",
        application_args=[
            "--gold-dir",    f"{BASE_DELTA}/gold",
            "--model-dir",   f"{MODEL_DIR}",
            "--metrics-dir", f"{BASE_DELTA}/metrics/{{ ds_nodash }}",
            "--model-type",  "rf",
            "--cv-folds",    "3",
        ],
        packages=PYSPARK_PKGS,
        executor_cores=4,
        executor_memory="8g",
        driver_memory="4g",
        name="medirisk-train-{{ ds_nodash }}",
    )

    batch_score_claims = SparkSubmitOperator(
        task_id="batch_score_claims",
        conn_id=SPARK_CONN_ID,
        application="pyspark/ml/batch_scoring.py",
        application_args=[
            "--gold-dir",    f"{BASE_DELTA}/gold",
            "--model-path",  f"{MODEL_DIR}/fraud_detection_rf",
            "--output-path", f"{BASE_DELTA}/gold/fraud_scores_snowpipe",
            "--threshold",   "0.5",
        ],
        packages=PYSPARK_PKGS,
        name="medirisk-score-{{ ds_nodash }}",
    )

    # ── Join after branch ──────────────────────────────────────────────────────
    ml_complete = EmptyOperator(
        task_id="ml_complete",
        trigger_rule=TriggerRule.ONE_SUCCESS,
    )

    # ── Snowflake: refresh Snowpipe ────────────────────────────────────────────
    snowpipe_refresh = SnowflakeOperator(
        task_id="snowpipe_refresh",
        snowflake_conn_id=SNOWFLAKE_CONN_ID,
        sql="""
            ALTER PIPE MEDIRISK.RAW.MEDIRISK_FRAUD_SCORES_PIPE REFRESH;
            ALTER PIPE MEDIRISK.RAW.MEDIRISK_PATIENTS_PIPE     REFRESH;
            ALTER PIPE MEDIRISK.RAW.MEDIRISK_PROVIDERS_PIPE    REFRESH;
        """,
    )

    # ── Snowflake: trigger CDC tasks ───────────────────────────────────────────
    trigger_cdc_tasks = SnowflakeOperator(
        task_id="trigger_cdc_tasks",
        snowflake_conn_id=SNOWFLAKE_CONN_ID,
        sql="""
            EXECUTE TASK MEDIRISK.SILVER.CDC_ORCHESTRATOR_TASK;
        """,
    )

    # ── Snowpark: risk scoring ─────────────────────────────────────────────────
    snowpark_risk_scoring = SnowflakeOperator(
        task_id="snowpark_risk_scoring",
        snowflake_conn_id=SNOWFLAKE_CONN_ID,
        sql="""
            CALL MEDIRISK.GOLD.RUN_RISK_SCORING();
            CALL MEDIRISK.GOLD.RUN_OUTLIER_DETECTION(3.0);
        """,
    )

    # ── Verify Dynamic Tables freshness ───────────────────────────────────────
    verify_dynamic_tables = SnowflakeOperator(
        task_id="verify_dynamic_tables",
        snowflake_conn_id=SNOWFLAKE_CONN_ID,
        sql="""
            -- Fail the task if any Dynamic Table hasn't refreshed in 60 minutes
            SELECT
                CASE WHEN COUNT(*) > 0
                THEN RAISE_ERROR('Dynamic Table refresh stale: ' || LISTAGG(name, ', '))
                END
            FROM (
                SELECT name
                FROM INFORMATION_SCHEMA.DYNAMIC_TABLES
                WHERE TABLE_SCHEMA = 'GOLD'
                  AND DATEDIFF('minute', LAST_COMPLETED_DEPENDENCY_REFRESH_TIME, CURRENT_TIMESTAMP()) > 60
            );
        """,
    )

    # ── Finish ─────────────────────────────────────────────────────────────────
    finish = EmptyOperator(
        task_id="finish",
        trigger_rule=TriggerRule.ALL_SUCCESS,
    )

    # ── DAG Wiring ─────────────────────────────────────────────────────────────
    (
        start
        >> bronze_ingest
        >> silver_transform
        >> gold_features
        >> branch_ml
        >> [train_fraud_model, batch_score_claims]
    )

    [train_fraud_model, batch_score_claims] >> ml_complete

    (
        ml_complete
        >> snowpipe_refresh
        >> trigger_cdc_tasks
        >> snowpark_risk_scoring
        >> verify_dynamic_tables
        >> finish
    )
