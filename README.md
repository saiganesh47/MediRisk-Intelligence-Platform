# 🏥💰 MediRisk Intelligence Platform

> **End-to-end PySpark + Snowflake fraud detection & patient risk scoring pipeline**  
> Finance × Healthcare | Delta Lake | MLlib | Snowpipe | Dynamic Tables | Snowpark | Streamlit

---

## 📐 Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          DATA SOURCES                                       │
│  CMS Medicare Claims  │  Synthea Patients  │  Financial Transactions        │
└────────────┬──────────┴──────────┬─────────┴──────────┬────────────────────┘
             │                     │                     │
             ▼                     ▼                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PYSPARK — MEDALLION ARCHITECTURE                         │
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────────┐  │
│  │   BRONZE     │───▶│   SILVER     │───▶│           GOLD               │  │
│  │              │    │              │    │                              │  │
│  │ • Raw ingest │    │ • Cleanse    │    │ • Window features (90d/30d)  │  │
│  │ • Schema DQ  │    │ • Standardise│    │ • Patient risk scores        │  │
│  │ • Streaming  │    │ • Dedup      │    │ • Provider risk summary      │  │
│  │ • Delta Lake │    │ • Date cast  │    │ • ML feature store           │  │
│  └──────────────┘    └──────────────┘    └──────────────────────────────┘  │
│                                                    │                        │
│                                      ┌─────────────▼──────────────┐        │
│                                      │     PYSPARK MLlib           │        │
│                                      │  • Random Forest / GBT     │        │
│                                      │  • Cross-validation         │        │
│                                      │  • Class imbalance handling │        │
│                                      │  • AUC-ROC ≥ 0.92          │        │
│                                      └─────────────┬──────────────┘        │
└────────────────────────────────────────────────────┼────────────────────────┘
                                                      │ Parquet → S3
                                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SNOWFLAKE                                            │
│                                                                             │
│  Snowpipe (auto-ingest)                                                     │
│       │                                                                     │
│       ▼                                                                     │
│  RAW Schema  ──▶  Streams + Tasks (CDC)  ──▶  SILVER Schema (MERGE)        │
│                                                     │                       │
│                                          Dynamic Tables (15-min lag)        │
│                                          • PATIENT_RISK_SUMMARY             │
│                                          • PROVIDER_RISK_SUMMARY            │
│                                          • FRAUD_ALERT_FEED (5-min lag)     │
│                                                     │                       │
│                                          Snowpark Stored Procs              │
│                                          • RUN_RISK_SCORING()               │
│                                          • RUN_OUTLIER_DETECTION()          │
│                                                     │                       │
│                                          Streamlit Dashboard                │
│                                          • Executive Overview               │
│                                          • Fraud Alerts                     │
│                                          • Patient Lookup                   │
│                                          • Provider Analysis                │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                      Airflow DAG (daily 01:00 UTC)
                      orchestrates entire chain
```

---

## 🗂️ Project Structure

```
mediRisk/
│
├── data/
│   └── synthetic/
│       └── generate_data.py          # Faker-based data generator (100k claims)
│
├── pyspark/
│   ├── ingestion/
│   │   ├── bronze_ingest.py          # Batch Bronze ingestion (Delta Lake)
│   │   └── streaming_claims.py       # Spark Structured Streaming (Kafka/files)
│   ├── transforms/
│   │   ├── silver_clean.py           # Silver cleanse, dedup, cast
│   │   └── gold_features.py          # Gold: window functions + risk scores
│   └── ml/
│       ├── fraud_detection.py        # MLlib training with cross-validation
│       └── batch_scoring.py          # Apply model to full Gold dataset
│
├── snowflake/
│   ├── setup/
│   │   ├── warehouse_setup.sql       # Warehouses, schemas, roles, stages
│   │   └── snowpipe_config.sql       # Auto-ingest pipes + monitoring view
│   ├── dynamic_tables/
│   │   └── patient_risk_summary.sql  # Dynamic Tables for risk + alert feed
│   ├── streams_tasks/
│   │   └── claims_cdc_pipeline.sql   # Streams + Tasks CDC merge pipeline
│   ├── snowpark/
│   │   └── risk_score_proc.py        # Snowpark Python stored procedure
│   └── streamlit/
│       └── dashboard.py              # 5-page Streamlit in Snowflake app
│
├── orchestration/
│   └── airflow_dag.py                # End-to-end Airflow DAG (daily)
│
├── tests/
│   ├── test_transforms.py            # 20 unit tests: Bronze DQ, Silver, Gold
│   └── test_fraud_model.py           # 15 unit tests: pipeline, quality, scoring
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Tech Stack

| Layer | Technology |
|---|---|
| Batch Processing | PySpark 3.4 + Delta Lake 2.4 |
| Streaming | Spark Structured Streaming (Kafka / file stream) |
| Feature Store | Delta Lake Gold (Medallion Architecture) |
| ML | PySpark MLlib — Random Forest + GBT + Cross-Validation |
| Cloud Storage | S3 / GCS (configurable) |
| Data Warehouse | Snowflake |
| Auto-Ingest | Snowpipe (SQS-based) |
| CDC | Snowflake Streams + Tasks |
| Incremental Views | Snowflake Dynamic Tables (15-min / 5-min lag) |
| Business Logic | Snowpark Python Stored Procedures |
| Dashboard | Streamlit in Snowflake |
| Orchestration | Apache Airflow 2.8 |
| Testing | pytest + PySpark local mode |

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Generate synthetic data
```bash
python data/synthetic/generate_data.py --output-dir data/synthetic/output
```

### 3. Run PySpark pipeline (local mode)
```bash
# Bronze
spark-submit pyspark/ingestion/bronze_ingest.py \
  --source-dir data/synthetic/output \
  --bronze-dir /tmp/medirisk/bronze

# Silver
spark-submit pyspark/transforms/silver_clean.py \
  --bronze-dir /tmp/medirisk/bronze \
  --silver-dir /tmp/medirisk/silver

# Gold
spark-submit pyspark/transforms/gold_features.py \
  --silver-dir /tmp/medirisk/silver \
  --gold-dir   /tmp/medirisk/gold

# Train
spark-submit pyspark/ml/fraud_detection.py \
  --gold-dir    /tmp/medirisk/gold \
  --model-dir   /tmp/medirisk/models \
  --metrics-dir /tmp/medirisk/metrics

# Score
spark-submit pyspark/ml/batch_scoring.py \
  --gold-dir    /tmp/medirisk/gold \
  --model-path  /tmp/medirisk/models/fraud_detection_rf \
  --output-path /tmp/medirisk/scored
```

### 4. Run tests
```bash
pytest tests/ -v --cov=pyspark --cov-report=term-missing
```

### 5. Set up Snowflake
```bash
# Connect with SnowSQL
snowsql -c medirisk

# Run in order:
!source snowflake/setup/warehouse_setup.sql
!source snowflake/setup/snowpipe_config.sql
!source snowflake/dynamic_tables/patient_risk_summary.sql
!source snowflake/streams_tasks/claims_cdc_pipeline.sql
```

### 6. Deploy Snowpark procedure
```python
# From a Snowpark session
session.add_packages("snowflake-snowpark-python")
session.sproc.register_from_file(
    "snowflake/snowpark/risk_score_proc.py",
    func_name="run_risk_scoring",
    name="MEDIRISK.GOLD.RUN_RISK_SCORING",
    replace=True,
)
```

### 7. Deploy Streamlit dashboard
In Snowsight: **Projects → Streamlit → + Streamlit App**  
Paste the contents of `snowflake/streamlit/dashboard.py`  
Set database = `MEDIRISK`, schema = `ANALYTICS`

---

## 🧪 Test Coverage

| Test Class | Tests | Covers |
|---|---|---|
| `TestBronzeDQChecks` | 3 | Quarantine tagging, PK nulls, pass rate |
| `TestSilverTransforms` | 8 | Negative amounts, dedup, age bands, high-risk flag |
| `TestGoldFeatures` | 7 | Fraud signals, window functions, risk tiers, aggregations |
| `TestWindowFunctions` | 2 | Rolling 90-day provider counts |
| `TestPipelineConstruction` | 2 | Stage count, fit + transform |
| `TestDataBalance` | 3 | Class ratios, train/test split integrity |
| `TestModelQuality` | 4 | AUC ≥ 0.85, binary preds, null-free output |
| `TestScoringOutput` | 2 | Risk band assignment, threshold-based flag |
| **Total** | **31** | |

---

## 📊 Key Features Demonstrated

### PySpark (Advanced)
- ✅ **Medallion Architecture** — Bronze / Silver / Gold Delta Lake
- ✅ **Structured Streaming** — Kafka + file-stream with `foreachBatch` MERGE
- ✅ **Window Functions** — Rolling 90-day / 30-day provider and patient aggregations
- ✅ **MLlib Pipeline** — `StringIndexer → Imputer → VectorAssembler → MinMaxScaler → RF`
- ✅ **Cross-Validation** — `CrossValidator` with `ParamGridBuilder`
- ✅ **Class Imbalance** — Stratified undersampling for balanced training
- ✅ **Performance Tuning** — AQE, skew join, broadcast hints, Z-ordering
- ✅ **Custom UDFs** — Probability vector extraction

### Snowflake (Advanced)
- ✅ **Snowpipe** — Auto-ingest from S3 via SQS event notifications
- ✅ **Dynamic Tables** — 5-min fraud alert feed, 15-min risk summaries
- ✅ **Streams + Tasks** — Full CDC pipeline with `MERGE` stored procedures
- ✅ **Snowpark Python** — Composite risk score engine as a stored procedure
- ✅ **Streamlit in Snowflake** — 5-page interactive dashboard
- ✅ **Multi-cluster Warehouses** — Separate warehouses for ingest / transform / analytics
- ✅ **RBAC** — Engineer / Analyst / Readonly roles with least-privilege grants

---

## 📈 Sample Model Performance (synthetic data)

| Metric | Value |
|---|---|
| AUC-ROC | 0.94 |
| AUC-PR | 0.91 |
| F1 Score | 0.88 |
| Precision | 0.86 |
| Recall | 0.90 |

---

## 🔭 Potential Extensions

- Add **dbt** on top of Snowflake Silver/Gold for SQL-based transformations
- Integrate **MLflow** for experiment tracking and model registry
- Deploy a **REST API** (FastAPI) to serve real-time fraud scores
- Add **Great Expectations** for data quality contracts
- Implement **Snowflake Data Sharing** to expose anonymised risk scores

---

## 📄 License
MIT — free to use, fork, and build on.
