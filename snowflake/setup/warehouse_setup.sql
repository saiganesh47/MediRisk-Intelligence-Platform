-- =============================================================================
-- MediRisk Intelligence Platform
-- Snowflake Setup — Warehouses, Databases, Schemas, Roles, Storage Integration
-- =============================================================================

USE ROLE SYSADMIN;

-- ─── Warehouses ───────────────────────────────────────────────────────────────

CREATE WAREHOUSE IF NOT EXISTS MEDIRISK_INGEST_WH
    WAREHOUSE_SIZE   = 'MEDIUM'
    AUTO_SUSPEND     = 120
    AUTO_RESUME      = TRUE
    MAX_CLUSTER_COUNT = 3
    MIN_CLUSTER_COUNT = 1
    SCALING_POLICY   = 'ECONOMY'
    COMMENT          = 'Used by Snowpipe and batch ingestion jobs';

CREATE WAREHOUSE IF NOT EXISTS MEDIRISK_TRANSFORM_WH
    WAREHOUSE_SIZE   = 'LARGE'
    AUTO_SUSPEND     = 300
    AUTO_RESUME      = TRUE
    MAX_CLUSTER_COUNT = 4
    MIN_CLUSTER_COUNT = 1
    SCALING_POLICY   = 'STANDARD'
    COMMENT          = 'Used for Dynamic Tables, Stream+Task processing';

CREATE WAREHOUSE IF NOT EXISTS MEDIRISK_ANALYTICS_WH
    WAREHOUSE_SIZE   = 'SMALL'
    AUTO_SUSPEND     = 60
    AUTO_RESUME      = TRUE
    COMMENT          = 'Used by Streamlit dashboard and ad-hoc analysts';


-- ─── Database & Schemas ───────────────────────────────────────────────────────

CREATE DATABASE IF NOT EXISTS MEDIRISK;

CREATE SCHEMA IF NOT EXISTS MEDIRISK.RAW
    COMMENT = 'Landing zone — raw data from Snowpipe / PySpark export';

CREATE SCHEMA IF NOT EXISTS MEDIRISK.SILVER
    COMMENT = 'Cleaned, conformed tables aligned with PySpark Silver';

CREATE SCHEMA IF NOT EXISTS MEDIRISK.GOLD
    COMMENT = 'Aggregated risk scores, feature store, ML outputs';

CREATE SCHEMA IF NOT EXISTS MEDIRISK.ANALYTICS
    COMMENT = 'Reporting views, KPI tables, Streamlit sources';

CREATE SCHEMA IF NOT EXISTS MEDIRISK.ML
    COMMENT = 'Fraud model scores, batch predictions';


-- ─── Roles ────────────────────────────────────────────────────────────────────

USE ROLE USERADMIN;

CREATE ROLE IF NOT EXISTS MEDIRISK_ENGINEER;
CREATE ROLE IF NOT EXISTS MEDIRISK_ANALYST;
CREATE ROLE IF NOT EXISTS MEDIRISK_READONLY;

-- Grant privileges
GRANT USAGE  ON WAREHOUSE MEDIRISK_INGEST_WH    TO ROLE MEDIRISK_ENGINEER;
GRANT USAGE  ON WAREHOUSE MEDIRISK_TRANSFORM_WH TO ROLE MEDIRISK_ENGINEER;
GRANT USAGE  ON WAREHOUSE MEDIRISK_ANALYTICS_WH TO ROLE MEDIRISK_ANALYST;
GRANT USAGE  ON WAREHOUSE MEDIRISK_ANALYTICS_WH TO ROLE MEDIRISK_READONLY;

GRANT ALL    ON DATABASE MEDIRISK               TO ROLE MEDIRISK_ENGINEER;
GRANT USAGE  ON DATABASE MEDIRISK               TO ROLE MEDIRISK_ANALYST;
GRANT USAGE  ON DATABASE MEDIRISK               TO ROLE MEDIRISK_READONLY;

GRANT ALL    ON ALL SCHEMAS IN DATABASE MEDIRISK TO ROLE MEDIRISK_ENGINEER;
GRANT USAGE  ON SCHEMA MEDIRISK.GOLD             TO ROLE MEDIRISK_ANALYST;
GRANT USAGE  ON SCHEMA MEDIRISK.ANALYTICS        TO ROLE MEDIRISK_ANALYST;
GRANT SELECT ON ALL TABLES IN SCHEMA MEDIRISK.ANALYTICS TO ROLE MEDIRISK_READONLY;


-- ─── Storage Integration (S3) ─────────────────────────────────────────────────

USE ROLE ACCOUNTADMIN;

CREATE STORAGE INTEGRATION IF NOT EXISTS MEDIRISK_S3_INTEGRATION
    TYPE                      = EXTERNAL_STAGE
    STORAGE_PROVIDER          = 'S3'
    ENABLED                   = TRUE
    STORAGE_AWS_ROLE_ARN      = 'arn:aws:iam::123456789012:role/medirisk-snowflake-role'
    STORAGE_ALLOWED_LOCATIONS = ('s3://medirisk-delta/', 's3://medirisk-landing/');

-- After creation, run this to retrieve the AWS IAM values to update the trust policy:
-- DESC INTEGRATION MEDIRISK_S3_INTEGRATION;


-- ─── External Stages ──────────────────────────────────────────────────────────

USE ROLE MEDIRISK_ENGINEER;
USE SCHEMA MEDIRISK.RAW;

CREATE STAGE IF NOT EXISTS MEDIRISK_CLAIMS_STAGE
    URL                = 's3://medirisk-delta/gold/fraud_scores_snowpipe/'
    STORAGE_INTEGRATION = MEDIRISK_S3_INTEGRATION
    FILE_FORMAT        = (TYPE = 'PARQUET')
    COMMENT            = 'PySpark-scored claims landing zone';

CREATE STAGE IF NOT EXISTS MEDIRISK_PATIENTS_STAGE
    URL                = 's3://medirisk-delta/silver/patients/'
    STORAGE_INTEGRATION = MEDIRISK_S3_INTEGRATION
    FILE_FORMAT        = (TYPE = 'PARQUET')
    COMMENT            = 'PySpark Silver patients output';

CREATE STAGE IF NOT EXISTS MEDIRISK_PROVIDERS_STAGE
    URL                = 's3://medirisk-delta/silver/providers/'
    STORAGE_INTEGRATION = MEDIRISK_S3_INTEGRATION
    FILE_FORMAT        = (TYPE = 'PARQUET')
    COMMENT            = 'PySpark Silver providers output';


-- ─── Landing Tables (RAW schema) ──────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS MEDIRISK.RAW.FRAUD_SCORES_RAW (
    CLAIM_ID                VARCHAR(36)   NOT NULL,
    PATIENT_ID              VARCHAR(36)   NOT NULL,
    PROVIDER_ID             VARCHAR(36)   NOT NULL,
    SERVICE_DATE            DATE,
    BILLED_AMOUNT           FLOAT,
    DIAGNOSIS_CODE          VARCHAR(20),
    PROCEDURE_CODE          VARCHAR(20),
    FRAUD_PROBABILITY       FLOAT,
    PREDICTED_FRAUD_LABEL   FLOAT,
    FRAUD_FLAG              INTEGER,
    RISK_BAND               VARCHAR(10),
    SCORED_AT               TIMESTAMP_NTZ,
    _SNOWPIPE_LOADED_AT     TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

CREATE TABLE IF NOT EXISTS MEDIRISK.RAW.PATIENTS_RAW (
    PATIENT_ID          VARCHAR(36)  NOT NULL,
    FIRST_NAME          VARCHAR(100),
    LAST_NAME           VARCHAR(100),
    DOB                 DATE,
    GENDER              VARCHAR(5),
    ZIP_CODE            VARCHAR(10),
    STATE               VARCHAR(5),
    INSURANCE_TYPE      VARCHAR(50),
    INSURANCE_ID        VARCHAR(50),
    CHRONIC_CONDITIONS  INTEGER,
    AGE                 INTEGER,
    AGE_BAND            VARCHAR(10),
    HIGH_RISK_FLAG      BOOLEAN,
    _SNOWPIPE_LOADED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

CREATE TABLE IF NOT EXISTS MEDIRISK.RAW.PROVIDERS_RAW (
    PROVIDER_ID         VARCHAR(36)  NOT NULL,
    NPI                 VARCHAR(20),
    NAME                VARCHAR(200),
    SPECIALTY           VARCHAR(100),
    STATE               VARCHAR(5),
    ZIP_CODE            VARCHAR(10),
    IS_FLAGGED          BOOLEAN,
    LICENSE_VALID       BOOLEAN,
    RISK_TIER           VARCHAR(10),
    PROVIDER_RISK_SCORE FLOAT,
    _SNOWPIPE_LOADED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);
