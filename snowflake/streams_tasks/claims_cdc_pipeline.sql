-- =============================================================================
-- MediRisk Intelligence Platform
-- Streams + Tasks — CDC Pipeline for Claims Status Updates
-- Captures changes in RAW tables and propagates them downstream
-- =============================================================================

USE ROLE      MEDIRISK_ENGINEER;
USE DATABASE  MEDIRISK;
USE SCHEMA    MEDIRISK.RAW;
USE WAREHOUSE MEDIRISK_TRANSFORM_WH;


-- ─── Streams — capture CDC on RAW tables ──────────────────────────────────────

CREATE STREAM IF NOT EXISTS MEDIRISK.RAW.FRAUD_SCORES_STREAM
    ON TABLE MEDIRISK.RAW.FRAUD_SCORES_RAW
    APPEND_ONLY = FALSE    -- capture INSERTs, UPDATEs, DELETEs
    COMMENT     = 'CDC stream on fraud scores raw table';

CREATE STREAM IF NOT EXISTS MEDIRISK.RAW.PATIENTS_STREAM
    ON TABLE MEDIRISK.RAW.PATIENTS_RAW
    APPEND_ONLY = TRUE     -- patients are append-only
    COMMENT     = 'CDC stream on patients raw table';


-- ─── Silver Merge Table for Claims ────────────────────────────────────────────

USE SCHEMA MEDIRISK.SILVER;

CREATE TABLE IF NOT EXISTS MEDIRISK.SILVER.FRAUD_SCORES (
    CLAIM_ID              VARCHAR(36)   NOT NULL PRIMARY KEY,
    PATIENT_ID            VARCHAR(36)   NOT NULL,
    PROVIDER_ID           VARCHAR(36)   NOT NULL,
    SERVICE_DATE          DATE,
    BILLED_AMOUNT         FLOAT,
    DIAGNOSIS_CODE        VARCHAR(20),
    PROCEDURE_CODE        VARCHAR(20),
    FRAUD_PROBABILITY     FLOAT,
    FRAUD_FLAG            INTEGER,
    RISK_BAND             VARCHAR(10),
    SCORED_AT             TIMESTAMP_NTZ,
    _CDC_ACTION           VARCHAR(10),
    _SILVER_UPDATED_AT    TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

CREATE TABLE IF NOT EXISTS MEDIRISK.SILVER.PATIENTS (
    PATIENT_ID          VARCHAR(36) NOT NULL PRIMARY KEY,
    FIRST_NAME          VARCHAR(100),
    LAST_NAME           VARCHAR(100),
    DOB                 DATE,
    GENDER              VARCHAR(5),
    ZIP_CODE            VARCHAR(10),
    STATE               VARCHAR(5),
    INSURANCE_TYPE      VARCHAR(50),
    CHRONIC_CONDITIONS  INTEGER,
    AGE                 INTEGER,
    AGE_BAND            VARCHAR(10),
    HIGH_RISK_FLAG      BOOLEAN,
    _SILVER_UPDATED_AT  TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);


-- ─── Stored Procedure: Merge Fraud Scores from Stream ─────────────────────────

CREATE OR REPLACE PROCEDURE MEDIRISK.SILVER.MERGE_FRAUD_SCORES_FROM_STREAM()
RETURNS STRING
LANGUAGE SQL
EXECUTE AS CALLER
AS $$
BEGIN
    -- Only process if there are pending records
    LET row_count INTEGER := (SELECT COUNT(*) FROM MEDIRISK.RAW.FRAUD_SCORES_STREAM);
    
    IF (row_count = 0) THEN
        RETURN 'No new records in stream. Skipping.';
    END IF;

    MERGE INTO MEDIRISK.SILVER.FRAUD_SCORES AS target
    USING (
        SELECT
            CLAIM_ID,
            PATIENT_ID,
            PROVIDER_ID,
            SERVICE_DATE,
            BILLED_AMOUNT,
            DIAGNOSIS_CODE,
            PROCEDURE_CODE,
            FRAUD_PROBABILITY,
            FRAUD_FLAG,
            RISK_BAND,
            SCORED_AT,
            METADATA$ACTION    AS _cdc_action,
            METADATA$ISUPDATE  AS _is_update
        FROM MEDIRISK.RAW.FRAUD_SCORES_STREAM
        WHERE METADATA$ACTION != 'DELETE'
    ) AS source
    ON target.CLAIM_ID = source.CLAIM_ID
    WHEN MATCHED AND source._is_update THEN
        UPDATE SET
            FRAUD_PROBABILITY   = source.FRAUD_PROBABILITY,
            FRAUD_FLAG          = source.FRAUD_FLAG,
            RISK_BAND           = source.RISK_BAND,
            SCORED_AT           = source.SCORED_AT,
            _CDC_ACTION         = 'UPDATE',
            _SILVER_UPDATED_AT  = CURRENT_TIMESTAMP()
    WHEN NOT MATCHED THEN
        INSERT (
            CLAIM_ID, PATIENT_ID, PROVIDER_ID, SERVICE_DATE, BILLED_AMOUNT,
            DIAGNOSIS_CODE, PROCEDURE_CODE, FRAUD_PROBABILITY, FRAUD_FLAG,
            RISK_BAND, SCORED_AT, _CDC_ACTION
        )
        VALUES (
            source.CLAIM_ID, source.PATIENT_ID, source.PROVIDER_ID,
            source.SERVICE_DATE, source.BILLED_AMOUNT, source.DIAGNOSIS_CODE,
            source.PROCEDURE_CODE, source.FRAUD_PROBABILITY, source.FRAUD_FLAG,
            source.RISK_BAND, source.SCORED_AT, 'INSERT'
        );

    RETURN 'Merged ' || row_count || ' records from FRAUD_SCORES_STREAM.';
END;
$$;


-- ─── Stored Procedure: Merge Patients from Stream ─────────────────────────────

CREATE OR REPLACE PROCEDURE MEDIRISK.SILVER.MERGE_PATIENTS_FROM_STREAM()
RETURNS STRING
LANGUAGE SQL
EXECUTE AS CALLER
AS $$
BEGIN
    LET row_count INTEGER := (SELECT COUNT(*) FROM MEDIRISK.RAW.PATIENTS_STREAM);
    
    IF (row_count = 0) THEN
        RETURN 'No new records in stream. Skipping.';
    END IF;

    MERGE INTO MEDIRISK.SILVER.PATIENTS AS target
    USING (
        SELECT * FROM MEDIRISK.RAW.PATIENTS_STREAM
    ) AS source
    ON target.PATIENT_ID = source.PATIENT_ID
    WHEN MATCHED THEN
        UPDATE SET
            CHRONIC_CONDITIONS = source.CHRONIC_CONDITIONS,
            HIGH_RISK_FLAG     = source.HIGH_RISK_FLAG,
            _SILVER_UPDATED_AT = CURRENT_TIMESTAMP()
    WHEN NOT MATCHED THEN
        INSERT (
            PATIENT_ID, FIRST_NAME, LAST_NAME, DOB, GENDER, ZIP_CODE,
            STATE, INSURANCE_TYPE, CHRONIC_CONDITIONS, AGE, AGE_BAND, HIGH_RISK_FLAG
        )
        VALUES (
            source.PATIENT_ID, source.FIRST_NAME, source.LAST_NAME, source.DOB,
            source.GENDER, source.ZIP_CODE, source.STATE, source.INSURANCE_TYPE,
            source.CHRONIC_CONDITIONS, source.AGE, source.AGE_BAND, source.HIGH_RISK_FLAG
        );

    RETURN 'Merged ' || row_count || ' patient records.';
END;
$$;


-- ─── Tasks — schedule the merge procedures ────────────────────────────────────

-- Root task: runs every 10 minutes, checks stream health
CREATE OR REPLACE TASK MEDIRISK.SILVER.CDC_ORCHESTRATOR_TASK
    WAREHOUSE = MEDIRISK_TRANSFORM_WH
    SCHEDULE  = '10 MINUTE'
    COMMENT   = 'Orchestrates CDC merges from streams to Silver'
AS
    SELECT SYSTEM$TASK_DEPENDENTS_ENABLE('MEDIRISK.SILVER.CDC_ORCHESTRATOR_TASK');

-- Fraud scores merge task (child)
CREATE OR REPLACE TASK MEDIRISK.SILVER.MERGE_FRAUD_SCORES_TASK
    WAREHOUSE = MEDIRISK_TRANSFORM_WH
    AFTER     MEDIRISK.SILVER.CDC_ORCHESTRATOR_TASK
    WHEN      SYSTEM$STREAM_HAS_DATA('MEDIRISK.RAW.FRAUD_SCORES_STREAM')
AS
    CALL MEDIRISK.SILVER.MERGE_FRAUD_SCORES_FROM_STREAM();

-- Patients merge task (child)
CREATE OR REPLACE TASK MEDIRISK.SILVER.MERGE_PATIENTS_TASK
    WAREHOUSE = MEDIRISK_TRANSFORM_WH
    AFTER     MEDIRISK.SILVER.CDC_ORCHESTRATOR_TASK
    WHEN      SYSTEM$STREAM_HAS_DATA('MEDIRISK.RAW.PATIENTS_STREAM')
AS
    CALL MEDIRISK.SILVER.MERGE_PATIENTS_FROM_STREAM();

-- Resume tasks (they start in SUSPENDED state by default)
ALTER TASK MEDIRISK.SILVER.MERGE_FRAUD_SCORES_TASK RESUME;
ALTER TASK MEDIRISK.SILVER.MERGE_PATIENTS_TASK     RESUME;
ALTER TASK MEDIRISK.SILVER.CDC_ORCHESTRATOR_TASK   RESUME;


-- ─── Task History Monitor ─────────────────────────────────────────────────────

CREATE OR REPLACE VIEW MEDIRISK.ANALYTICS.TASK_RUN_HISTORY AS
SELECT
    NAME,
    STATE,
    SCHEDULED_TIME,
    QUERY_START_TIME,
    COMPLETED_TIME,
    DATEDIFF('second', QUERY_START_TIME, COMPLETED_TIME) AS duration_seconds,
    ERROR_CODE,
    ERROR_MESSAGE
FROM TABLE(INFORMATION_SCHEMA.TASK_HISTORY(
    SCHEDULED_TIME_RANGE_START => DATEADD('hour', -24, CURRENT_TIMESTAMP()),
    RESULT_LIMIT               => 200
))
ORDER BY SCHEDULED_TIME DESC;
