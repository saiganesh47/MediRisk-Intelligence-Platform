-- =============================================================================
-- MediRisk Intelligence Platform
-- Dynamic Tables — Auto-Refreshing Patient & Provider Risk Summaries
-- Uses Snowflake Dynamic Tables (GA since 2024) for incremental refresh
-- =============================================================================

USE ROLE      MEDIRISK_ENGINEER;
USE DATABASE  MEDIRISK;
USE SCHEMA    MEDIRISK.GOLD;
USE WAREHOUSE MEDIRISK_TRANSFORM_WH;


-- ─── Patient Risk Summary ─────────────────────────────────────────────────────

CREATE OR REPLACE DYNAMIC TABLE MEDIRISK.GOLD.PATIENT_RISK_SUMMARY
    TARGET_LAG      = '15 minutes'
    WAREHOUSE       = MEDIRISK_TRANSFORM_WH
    COMMENT         = 'Auto-refreshed patient risk profile combining claims, fraud scores, and demographics'
AS
WITH patient_claims AS (
    SELECT
        fs.PATIENT_ID,
        COUNT(fs.CLAIM_ID)                                           AS total_claims,
        SUM(fs.BILLED_AMOUNT)                                        AS lifetime_billed,
        AVG(fs.BILLED_AMOUNT)                                        AS avg_claim_billed,
        MAX(fs.BILLED_AMOUNT)                                        AS max_claim_billed,
        SUM(fs.FRAUD_FLAG)                                           AS fraud_claim_count,
        COUNT(DISTINCT fs.PROVIDER_ID)                               AS distinct_providers,
        COUNT(DISTINCT fs.PROCEDURE_CODE)                            AS distinct_procedures,
        AVG(fs.FRAUD_PROBABILITY)                                    AS avg_fraud_probability,
        MAX(fs.FRAUD_PROBABILITY)                                    AS max_fraud_probability,
        SUM(CASE WHEN fs.RISK_BAND = 'HIGH'   THEN 1 ELSE 0 END)    AS high_risk_claim_count,
        SUM(CASE WHEN fs.RISK_BAND = 'MEDIUM' THEN 1 ELSE 0 END)    AS medium_risk_claim_count,
        MAX(fs.SERVICE_DATE)                                         AS last_service_date,
        MAX(fs.SCORED_AT)                                            AS last_scored_at
    FROM MEDIRISK.RAW.FRAUD_SCORES_RAW fs
    GROUP BY 1
),
patient_30d AS (
    SELECT
        PATIENT_ID,
        COUNT(CLAIM_ID)             AS claims_last_30d,
        SUM(BILLED_AMOUNT)          AS billed_last_30d,
        COUNT(DISTINCT PROVIDER_ID) AS providers_last_30d
    FROM MEDIRISK.RAW.FRAUD_SCORES_RAW
    WHERE SERVICE_DATE >= DATEADD('day', -30, CURRENT_DATE())
    GROUP BY 1
)
SELECT
    p.PATIENT_ID,
    p.FIRST_NAME,
    p.LAST_NAME,
    p.DOB,
    p.AGE,
    p.AGE_BAND,
    p.GENDER,
    p.STATE,
    p.ZIP_CODE,
    p.INSURANCE_TYPE,
    p.CHRONIC_CONDITIONS,
    p.HIGH_RISK_FLAG,

    -- Claims stats
    COALESCE(pc.total_claims,            0)     AS total_claims,
    COALESCE(pc.lifetime_billed,         0)     AS lifetime_billed,
    COALESCE(pc.avg_claim_billed,        0)     AS avg_claim_billed,
    COALESCE(pc.max_claim_billed,        0)     AS max_claim_billed,
    COALESCE(pc.fraud_claim_count,       0)     AS fraud_claim_count,
    COALESCE(pc.distinct_providers,      0)     AS distinct_providers,
    COALESCE(pc.distinct_procedures,     0)     AS distinct_procedures,
    COALESCE(pc.avg_fraud_probability,   0)     AS avg_fraud_probability,
    COALESCE(pc.max_fraud_probability,   0)     AS max_fraud_probability,
    COALESCE(pc.high_risk_claim_count,   0)     AS high_risk_claim_count,
    COALESCE(pc.medium_risk_claim_count, 0)     AS medium_risk_claim_count,
    pc.last_service_date,
    pc.last_scored_at,

    -- 30-day rolling
    COALESCE(p30.claims_last_30d,    0)         AS claims_last_30d,
    COALESCE(p30.billed_last_30d,    0)         AS billed_last_30d,
    COALESCE(p30.providers_last_30d, 0)         AS providers_last_30d,

    -- Fraud rate
    CASE
        WHEN COALESCE(pc.total_claims, 0) = 0 THEN 0
        ELSE ROUND(pc.fraud_claim_count / pc.total_claims, 4)
    END AS fraud_claim_rate,

    -- Composite risk score (0–100)
    LEAST(100,
        ROUND(
            COALESCE(pc.avg_fraud_probability, 0) * 40 +
            COALESCE(p.chronic_conditions,     0) *  5 +
            CASE WHEN COALESCE(pc.distinct_providers, 0) > 5 THEN 15 ELSE 0 END +
            CASE WHEN p.high_risk_flag THEN 10 ELSE 0 END +
            COALESCE(pc.high_risk_claim_count, 0) * 2,
        2)
    ) AS composite_risk_score,

    -- Risk tier
    CASE
        WHEN LEAST(100, ROUND(
            COALESCE(pc.avg_fraud_probability, 0) * 40 +
            COALESCE(p.chronic_conditions,     0) *  5 +
            CASE WHEN COALESCE(pc.distinct_providers, 0) > 5 THEN 15 ELSE 0 END +
            CASE WHEN p.high_risk_flag THEN 10 ELSE 0 END +
            COALESCE(pc.high_risk_claim_count, 0) * 2, 2)) >= 70 THEN 'HIGH'
        WHEN LEAST(100, ROUND(
            COALESCE(pc.avg_fraud_probability, 0) * 40 +
            COALESCE(p.chronic_conditions,     0) *  5 +
            CASE WHEN COALESCE(pc.distinct_providers, 0) > 5 THEN 15 ELSE 0 END +
            CASE WHEN p.high_risk_flag THEN 10 ELSE 0 END +
            COALESCE(pc.high_risk_claim_count, 0) * 2, 2)) >= 40 THEN 'MEDIUM'
        ELSE 'LOW'
    END AS risk_tier,

    CURRENT_TIMESTAMP() AS _last_refreshed_at

FROM MEDIRISK.RAW.PATIENTS_RAW    p
LEFT JOIN patient_claims           pc  ON p.PATIENT_ID = pc.PATIENT_ID
LEFT JOIN patient_30d              p30 ON p.PATIENT_ID = p30.PATIENT_ID;


-- ─── Provider Risk Summary ────────────────────────────────────────────────────

CREATE OR REPLACE DYNAMIC TABLE MEDIRISK.GOLD.PROVIDER_RISK_SUMMARY
    TARGET_LAG = '15 minutes'
    WAREHOUSE  = MEDIRISK_TRANSFORM_WH
    COMMENT    = 'Auto-refreshed provider risk profile with fraud statistics'
AS
SELECT
    pv.PROVIDER_ID,
    pv.NPI,
    pv.NAME            AS provider_name,
    pv.SPECIALTY,
    pv.STATE,
    pv.IS_FLAGGED,
    pv.LICENSE_VALID,
    pv.RISK_TIER       AS admin_risk_tier,
    pv.PROVIDER_RISK_SCORE AS admin_risk_score,

    COUNT(fs.CLAIM_ID)                                        AS total_claims,
    SUM(fs.BILLED_AMOUNT)                                     AS total_billed,
    AVG(fs.BILLED_AMOUNT)                                     AS avg_billed,
    STDDEV(fs.BILLED_AMOUNT)                                  AS stddev_billed,
    SUM(fs.FRAUD_FLAG)                                        AS fraud_claim_count,
    COUNT(DISTINCT fs.PATIENT_ID)                             AS unique_patients,
    AVG(fs.FRAUD_PROBABILITY)                                 AS avg_fraud_probability,
    MAX(fs.FRAUD_PROBABILITY)                                 AS max_fraud_probability,
    SUM(CASE WHEN fs.RISK_BAND = 'HIGH' THEN 1 ELSE 0 END)   AS high_risk_claims,
    MAX(fs.SERVICE_DATE)                                      AS last_service_date,

    CASE
        WHEN COUNT(fs.CLAIM_ID) = 0 THEN 0
        ELSE ROUND(SUM(fs.FRAUD_FLAG) / COUNT(fs.CLAIM_ID), 4)
    END AS fraud_rate,

    LEAST(100, ROUND(
        COALESCE(AVG(fs.FRAUD_PROBABILITY), 0) * 50 +
        CASE WHEN pv.IS_FLAGGED       THEN 30 ELSE 0 END +
        CASE WHEN NOT pv.LICENSE_VALID THEN 20 ELSE 0 END,
    2)) AS computed_risk_score,

    CURRENT_TIMESTAMP() AS _last_refreshed_at

FROM MEDIRISK.RAW.PROVIDERS_RAW   pv
LEFT JOIN MEDIRISK.RAW.FRAUD_SCORES_RAW fs ON pv.PROVIDER_ID = fs.PROVIDER_ID
GROUP BY 1, 2, 3, 4, 5, 6, 7, 8, 9;


-- ─── Fraud Alert Feed (high-risk claims only) ─────────────────────────────────

CREATE OR REPLACE DYNAMIC TABLE MEDIRISK.GOLD.FRAUD_ALERT_FEED
    TARGET_LAG = '5 minutes'
    WAREHOUSE  = MEDIRISK_TRANSFORM_WH
    COMMENT    = 'Near-real-time fraud alerts for HIGH risk claims'
AS
SELECT
    fs.CLAIM_ID,
    fs.PATIENT_ID,
    fs.PROVIDER_ID,
    p.FIRST_NAME || ' ' || p.LAST_NAME  AS patient_name,
    pv.NAME                             AS provider_name,
    pv.SPECIALTY,
    fs.SERVICE_DATE,
    fs.BILLED_AMOUNT,
    fs.DIAGNOSIS_CODE,
    fs.PROCEDURE_CODE,
    fs.FRAUD_PROBABILITY,
    fs.RISK_BAND,
    fs.SCORED_AT,
    pv.IS_FLAGGED                       AS provider_flagged,
    p.HIGH_RISK_FLAG                    AS patient_high_risk,
    CURRENT_TIMESTAMP()                 AS _alert_generated_at
FROM MEDIRISK.RAW.FRAUD_SCORES_RAW    fs
LEFT JOIN MEDIRISK.RAW.PATIENTS_RAW   p  ON fs.PATIENT_ID  = p.PATIENT_ID
LEFT JOIN MEDIRISK.RAW.PROVIDERS_RAW  pv ON fs.PROVIDER_ID = pv.PROVIDER_ID
WHERE fs.RISK_BAND = 'HIGH'
  AND fs.SCORED_AT >= DATEADD('day', -7, CURRENT_TIMESTAMP())
ORDER BY fs.FRAUD_PROBABILITY DESC;


-- ─── Monitor Dynamic Table refresh lag ───────────────────────────────────────
-- SELECT name, target_lag, scheduling_state, last_suspended_on
-- FROM INFORMATION_SCHEMA.DYNAMIC_TABLES
-- WHERE TABLE_SCHEMA = 'GOLD';
