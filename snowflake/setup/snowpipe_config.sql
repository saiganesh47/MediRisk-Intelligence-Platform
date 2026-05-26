-- =============================================================================
-- MediRisk Intelligence Platform
-- Snowpipe Configuration — Auto-Ingest from S3 via SQS Notifications
-- =============================================================================

USE ROLE     MEDIRISK_ENGINEER;
USE DATABASE MEDIRISK;
USE SCHEMA   MEDIRISK.RAW;
USE WAREHOUSE MEDIRISK_INGEST_WH;


-- ─── Pipes ────────────────────────────────────────────────────────────────────

-- Fraud Scores pipe (PySpark ML output → RAW)
CREATE PIPE IF NOT EXISTS MEDIRISK_FRAUD_SCORES_PIPE
    AUTO_INGEST = TRUE
    COMMENT     = 'Ingest PySpark ML fraud scores from S3'
AS
COPY INTO MEDIRISK.RAW.FRAUD_SCORES_RAW (
    CLAIM_ID,
    PATIENT_ID,
    PROVIDER_ID,
    SERVICE_DATE,
    BILLED_AMOUNT,
    DIAGNOSIS_CODE,
    PROCEDURE_CODE,
    FRAUD_PROBABILITY,
    PREDICTED_FRAUD_LABEL,
    FRAUD_FLAG,
    RISK_BAND,
    SCORED_AT
)
FROM (
    SELECT
        $1:claim_id::VARCHAR(36),
        $1:patient_id::VARCHAR(36),
        $1:provider_id::VARCHAR(36),
        $1:service_date::DATE,
        $1:billed_amount::FLOAT,
        $1:diagnosis_code::VARCHAR(20),
        $1:procedure_code::VARCHAR(20),
        $1:fraud_probability::FLOAT,
        $1:predicted_fraud_label::FLOAT,
        $1:fraud_flag::INTEGER,
        $1:risk_band::VARCHAR(10),
        $1:scored_at::TIMESTAMP_NTZ
    FROM @MEDIRISK_CLAIMS_STAGE
)
FILE_FORMAT = (TYPE = 'PARQUET');


-- Patients pipe
CREATE PIPE IF NOT EXISTS MEDIRISK_PATIENTS_PIPE
    AUTO_INGEST = TRUE
    COMMENT     = 'Ingest PySpark Silver patients from S3'
AS
COPY INTO MEDIRISK.RAW.PATIENTS_RAW (
    PATIENT_ID, FIRST_NAME, LAST_NAME, DOB, GENDER,
    ZIP_CODE, STATE, INSURANCE_TYPE, INSURANCE_ID,
    CHRONIC_CONDITIONS, AGE, AGE_BAND, HIGH_RISK_FLAG
)
FROM (
    SELECT
        $1:patient_id::VARCHAR(36),
        $1:first_name::VARCHAR(100),
        $1:last_name::VARCHAR(100),
        $1:dob::DATE,
        $1:gender::VARCHAR(5),
        $1:zip_code::VARCHAR(10),
        $1:state::VARCHAR(5),
        $1:insurance_type::VARCHAR(50),
        $1:insurance_id::VARCHAR(50),
        $1:chronic_conditions::INTEGER,
        $1:age::INTEGER,
        $1:age_band::VARCHAR(10),
        $1:high_risk_flag::BOOLEAN
    FROM @MEDIRISK_PATIENTS_STAGE
)
FILE_FORMAT = (TYPE = 'PARQUET');


-- Providers pipe
CREATE PIPE IF NOT EXISTS MEDIRISK_PROVIDERS_PIPE
    AUTO_INGEST = TRUE
    COMMENT     = 'Ingest PySpark Silver providers from S3'
AS
COPY INTO MEDIRISK.RAW.PROVIDERS_RAW (
    PROVIDER_ID, NPI, NAME, SPECIALTY, STATE,
    ZIP_CODE, IS_FLAGGED, LICENSE_VALID, RISK_TIER, PROVIDER_RISK_SCORE
)
FROM (
    SELECT
        $1:provider_id::VARCHAR(36),
        $1:npi::VARCHAR(20),
        $1:name::VARCHAR(200),
        $1:specialty::VARCHAR(100),
        $1:state::VARCHAR(5),
        $1:zip_code::VARCHAR(10),
        $1:is_flagged::BOOLEAN,
        $1:license_valid::BOOLEAN,
        $1:risk_tier::VARCHAR(10),
        $1:provider_risk_score::FLOAT
    FROM @MEDIRISK_PROVIDERS_STAGE
)
FILE_FORMAT = (TYPE = 'PARQUET');


-- ─── Post-creation: retrieve SQS ARN for S3 Event Notification ───────────────
-- Run the following and copy the notification_channel value into your S3 bucket's
-- Event Notifications config (Event type: s3:ObjectCreated:*)

SHOW PIPES LIKE 'MEDIRISK_%_PIPE';

-- To check pipe status and recent loads:
-- SELECT SYSTEM$PIPE_STATUS('MEDIRISK_FRAUD_SCORES_PIPE');
-- SELECT * FROM TABLE(INFORMATION_SCHEMA.COPY_HISTORY(
--     TABLE_NAME => 'FRAUD_SCORES_RAW',
--     START_TIME => DATEADD('hours', -1, CURRENT_TIMESTAMP())
-- ));


-- ─── Pipe Monitoring View ─────────────────────────────────────────────────────

CREATE OR REPLACE VIEW MEDIRISK.ANALYTICS.PIPE_HEALTH AS
SELECT
    'FRAUD_SCORES'  AS pipe_name,
    PARSE_JSON(SYSTEM$PIPE_STATUS('MEDIRISK_FRAUD_SCORES_PIPE'))['executionState']::STRING AS execution_state,
    PARSE_JSON(SYSTEM$PIPE_STATUS('MEDIRISK_FRAUD_SCORES_PIPE'))['pendingFileCount']::INTEGER AS pending_files,
    PARSE_JSON(SYSTEM$PIPE_STATUS('MEDIRISK_FRAUD_SCORES_PIPE'))['lastIngestedTimestamp']::TIMESTAMP_NTZ AS last_ingested_at
UNION ALL
SELECT
    'PATIENTS',
    PARSE_JSON(SYSTEM$PIPE_STATUS('MEDIRISK_PATIENTS_PIPE'))['executionState']::STRING,
    PARSE_JSON(SYSTEM$PIPE_STATUS('MEDIRISK_PATIENTS_PIPE'))['pendingFileCount']::INTEGER,
    PARSE_JSON(SYSTEM$PIPE_STATUS('MEDIRISK_PATIENTS_PIPE'))['lastIngestedTimestamp']::TIMESTAMP_NTZ
UNION ALL
SELECT
    'PROVIDERS',
    PARSE_JSON(SYSTEM$PIPE_STATUS('MEDIRISK_PROVIDERS_PIPE'))['executionState']::STRING,
    PARSE_JSON(SYSTEM$PIPE_STATUS('MEDIRISK_PROVIDERS_PIPE'))['pendingFileCount']::INTEGER,
    PARSE_JSON(SYSTEM$PIPE_STATUS('MEDIRISK_PROVIDERS_PIPE'))['lastIngestedTimestamp']::TIMESTAMP_NTZ;
