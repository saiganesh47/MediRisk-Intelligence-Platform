"""
MediRisk Intelligence Platform
Streamlit in Snowflake — Executive Fraud & Risk Dashboard
Deploy directly inside Snowflake via:
    Snowsight → Projects → Streamlit → + Streamlit App
    (paste this file and set the database/schema to MEDIRISK.ANALYTICS)
"""

import json

import altair as alt
import pandas as pd
import streamlit as st
from snowflake.snowpark.context import get_active_session

# ─── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="MediRisk — Fraud & Risk Intelligence",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Session ──────────────────────────────────────────────────────────────────

session = get_active_session()


# ─── Helpers ──────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_kpis() -> dict:
    row = session.sql("""
        SELECT
            COUNT(*)                                           AS total_patients,
            SUM(CASE WHEN RISK_TIER = 'HIGH'   THEN 1 END)    AS high_risk_patients,
            SUM(CASE WHEN RISK_TIER = 'MEDIUM' THEN 1 END)    AS medium_risk_patients,
            ROUND(AVG(COMPOSITE_RISK_SCORE), 2)                AS avg_risk_score,
            ROUND(AVG(AVG_FRAUD_PROBABILITY) * 100, 2)         AS avg_fraud_prob_pct
        FROM MEDIRISK.GOLD.PATIENT_RISK_SCORES_FINAL
    """).collect()[0]
    return row.as_dict()


@st.cache_data(ttl=300)
def load_fraud_alerts(limit: int = 200) -> pd.DataFrame:
    return session.sql(f"""
        SELECT
            CLAIM_ID, PATIENT_NAME, PROVIDER_NAME, SPECIALTY,
            SERVICE_DATE, BILLED_AMOUNT, FRAUD_PROBABILITY,
            RISK_BAND, PROVIDER_FLAGGED, PATIENT_HIGH_RISK,
            SCORED_AT
        FROM MEDIRISK.GOLD.FRAUD_ALERT_FEED
        ORDER BY FRAUD_PROBABILITY DESC
        LIMIT {limit}
    """).to_pandas()


@st.cache_data(ttl=300)
def load_risk_distribution() -> pd.DataFrame:
    return session.sql("""
        SELECT RISK_TIER, COUNT(*) AS patient_count
        FROM MEDIRISK.GOLD.PATIENT_RISK_SCORES_FINAL
        GROUP BY 1
        ORDER BY 1
    """).to_pandas()


@st.cache_data(ttl=300)
def load_fraud_by_specialty() -> pd.DataFrame:
    return session.sql("""
        SELECT
            pv.SPECIALTY,
            COUNT(fs.CLAIM_ID)                          AS total_claims,
            SUM(fs.FRAUD_FLAG)                          AS fraud_claims,
            ROUND(AVG(fs.FRAUD_PROBABILITY) * 100, 2)   AS avg_fraud_prob_pct,
            ROUND(SUM(fs.BILLED_AMOUNT) / 1e6, 2)       AS total_billed_millions
        FROM MEDIRISK.SILVER.FRAUD_SCORES           fs
        JOIN MEDIRISK.RAW.PROVIDERS_RAW             pv ON fs.PROVIDER_ID = pv.PROVIDER_ID
        GROUP BY 1
        ORDER BY avg_fraud_prob_pct DESC
        LIMIT 15
    """).to_pandas()


@st.cache_data(ttl=300)
def load_monthly_trend() -> pd.DataFrame:
    return session.sql("""
        SELECT
            DATE_TRUNC('month', SERVICE_DATE)        AS month,
            COUNT(CLAIM_ID)                          AS total_claims,
            SUM(FRAUD_FLAG)                          AS fraud_claims,
            ROUND(SUM(BILLED_AMOUNT) / 1e6, 2)       AS billed_millions
        FROM MEDIRISK.SILVER.FRAUD_SCORES
        WHERE SERVICE_DATE >= DATEADD('year', -2, CURRENT_DATE())
        GROUP BY 1
        ORDER BY 1
    """).to_pandas()


@st.cache_data(ttl=600)
def load_provider_outliers() -> pd.DataFrame:
    return session.sql("""
        SELECT
            PROVIDER_ID, SPECIALTY,
            ROUND(avg_billed, 2)        AS avg_billed,
            ROUND(Z_SCORE, 2)           AS z_score,
            total_claims,
            IS_FLAGGED
        FROM MEDIRISK.GOLD.PROVIDER_OUTLIERS
        ORDER BY Z_SCORE DESC
        LIMIT 50
    """).to_pandas()


@st.cache_data(ttl=300)
def search_patient(patient_id: str) -> pd.DataFrame:
    return session.sql(f"""
        SELECT *
        FROM MEDIRISK.GOLD.PATIENT_RISK_SCORES_FINAL
        WHERE PATIENT_ID = '{patient_id}'
    """).to_pandas()


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://via.placeholder.com/200x60?text=MediRisk", width=200)
    st.title("MediRisk")
    st.caption("Fraud & Risk Intelligence Platform")
    st.divider()

    page = st.radio(
        "Navigation",
        ["📊 Executive Overview", "🚨 Fraud Alerts", "👤 Patient Lookup",
         "🏥 Provider Analysis", "📈 Trend Analysis"],
        label_visibility="collapsed",
    )

    st.divider()
    risk_filter = st.multiselect(
        "Filter by Risk Tier",
        ["HIGH", "MEDIUM", "LOW"],
        default=["HIGH", "MEDIUM", "LOW"],
    )
    st.caption(f"Last refreshed: {pd.Timestamp.now().strftime('%H:%M:%S')}")
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()


# ─── Page: Executive Overview ─────────────────────────────────────────────────

if page == "📊 Executive Overview":
    st.title("📊 Executive Overview")
    st.markdown("Real-time fraud risk intelligence across all patients and providers.")

    kpis = load_kpis()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Patients",    f"{kpis.get('TOTAL_PATIENTS', 0):,}")
    c2.metric("🔴 High Risk",      f"{kpis.get('HIGH_RISK_PATIENTS', 0):,}",
              delta=f"{kpis.get('HIGH_RISK_PATIENTS', 0)/max(kpis.get('TOTAL_PATIENTS',1),1)*100:.1f}%")
    c3.metric("🟡 Medium Risk",    f"{kpis.get('MEDIUM_RISK_PATIENTS', 0):,}")
    c4.metric("Avg Risk Score",    f"{kpis.get('AVG_RISK_SCORE', 0):.1f}/100")
    c5.metric("Avg Fraud Prob",    f"{kpis.get('AVG_FRAUD_PROB_PCT', 0):.1f}%")

    st.divider()

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Risk Tier Distribution")
        dist_df = load_risk_distribution()
        colors  = {"HIGH": "#ef4444", "MEDIUM": "#f59e0b", "LOW": "#22c55e"}
        chart = (
            alt.Chart(dist_df)
            .mark_arc(innerRadius=60)
            .encode(
                theta=alt.Theta("patient_count:Q"),
                color=alt.Color("RISK_TIER:N",
                                scale=alt.Scale(
                                    domain=list(colors.keys()),
                                    range=list(colors.values()),
                                )),
                tooltip=["RISK_TIER", "patient_count"],
            )
            .properties(height=320)
        )
        st.altair_chart(chart, use_container_width=True)

    with col_right:
        st.subheader("Fraud Probability by Specialty")
        spec_df = load_fraud_by_specialty().head(10)
        chart = (
            alt.Chart(spec_df)
            .mark_bar(color="#6366f1")
            .encode(
                x=alt.X("avg_fraud_prob_pct:Q", title="Avg Fraud Probability (%)"),
                y=alt.Y("SPECIALTY:N", sort="-x"),
                tooltip=["SPECIALTY", "avg_fraud_prob_pct", "total_claims", "fraud_claims"],
            )
            .properties(height=320)
        )
        st.altair_chart(chart, use_container_width=True)


# ─── Page: Fraud Alerts ───────────────────────────────────────────────────────

elif page == "🚨 Fraud Alerts":
    st.title("🚨 Fraud Alerts")
    st.markdown("High-risk claims from the last 7 days, ranked by fraud probability.")

    alerts_df = load_fraud_alerts(500)

    # Filters
    col1, col2, col3 = st.columns(3)
    prob_threshold = col1.slider("Min Fraud Probability", 0.0, 1.0, 0.7, 0.01)
    show_flagged   = col2.checkbox("Flagged Providers Only", value=False)
    show_high_risk = col3.checkbox("High-Risk Patients Only", value=False)

    filtered = alerts_df[alerts_df["FRAUD_PROBABILITY"] >= prob_threshold]
    if show_flagged:
        filtered = filtered[filtered["PROVIDER_FLAGGED"] == True]
    if show_high_risk:
        filtered = filtered[filtered["PATIENT_HIGH_RISK"] == True]

    st.metric("Matching Alerts", len(filtered))

    # Color-code by risk band
    def color_risk(val):
        colors = {"HIGH": "background-color: #fee2e2", "MEDIUM": "background-color: #fef9c3"}
        return colors.get(val, "")

    display_cols = [
        "PATIENT_NAME", "PROVIDER_NAME", "SPECIALTY", "SERVICE_DATE",
        "BILLED_AMOUNT", "FRAUD_PROBABILITY", "RISK_BAND", "SCORED_AT",
    ]
    if set(display_cols).issubset(filtered.columns):
        styled = (
            filtered[display_cols]
            .style
            .applymap(color_risk, subset=["RISK_BAND"])
            .format({
                "BILLED_AMOUNT":      "${:,.2f}",
                "FRAUD_PROBABILITY":  "{:.2%}",
            })
        )
        st.dataframe(styled, use_container_width=True, height=500)
    else:
        st.dataframe(filtered, use_container_width=True, height=500)


# ─── Page: Patient Lookup ─────────────────────────────────────────────────────

elif page == "👤 Patient Lookup":
    st.title("👤 Patient Risk Lookup")

    patient_id = st.text_input("Enter Patient ID (UUID)", placeholder="e.g. 123e4567-e89b-12d3-a456-426614174000")

    if patient_id:
        result = search_patient(patient_id.strip())
        if result.empty:
            st.warning("No patient found with that ID.")
        else:
            row = result.iloc[0]

            c1, c2, c3 = st.columns(3)
            c1.metric("Risk Score",       f"{row.get('COMPOSITE_RISK_SCORE', 0):.1f}/100")
            c2.metric("Risk Tier",        row.get("RISK_TIER", "N/A"))
            c3.metric("Fraud Claims",     int(row.get("FRAUD_CLAIM_COUNT", 0)))

            st.divider()
            st.subheader("Patient Profile")
            st.json({
                "Name":               f"{row.get('FIRST_NAME')} {row.get('LAST_NAME')}",
                "Age":                int(row.get("AGE", 0)),
                "Age Band":           row.get("AGE_BAND"),
                "State":              row.get("STATE"),
                "Insurance":          row.get("INSURANCE_TYPE"),
                "Chronic Conditions": int(row.get("CHRONIC_CONDITIONS", 0)),
                "High Risk Flag":     bool(row.get("HIGH_RISK_FLAG", False)),
                "Total Claims":       int(row.get("TOTAL_CLAIMS", 0)),
                "Lifetime Billed":    f"${row.get('LIFETIME_BILLED', 0):,.2f}",
                "Fraud Claim Rate":   f"{row.get('FRAUD_CLAIM_RATE', 0):.2%}",
            })


# ─── Page: Provider Analysis ──────────────────────────────────────────────────

elif page == "🏥 Provider Analysis":
    st.title("🏥 Provider Billing Outlier Analysis")

    outliers_df = load_provider_outliers()

    if outliers_df.empty:
        st.info("No outliers detected. Run CALL MEDIRISK.GOLD.RUN_OUTLIER_DETECTION(3.0) to refresh.")
    else:
        st.metric("Flagged Outliers", len(outliers_df))

        chart = (
            alt.Chart(outliers_df)
            .mark_point(filled=True, size=100)
            .encode(
                x=alt.X("total_claims:Q", title="Total Claims"),
                y=alt.Y("avg_billed:Q",   title="Avg Billed Amount ($)"),
                color=alt.Color("z_score:Q",
                                scale=alt.Scale(scheme="reds"),
                                title="Z-Score"),
                shape=alt.condition(
                    alt.datum.IS_FLAGGED, alt.value("triangle-up"), alt.value("circle")
                ),
                tooltip=["PROVIDER_ID", "SPECIALTY", "avg_billed", "z_score", "total_claims", "IS_FLAGGED"],
            )
            .properties(height=400, title="Provider Billing Outliers (bubble = z-score)")
        )
        st.altair_chart(chart, use_container_width=True)
        st.dataframe(outliers_df, use_container_width=True)


# ─── Page: Trend Analysis ─────────────────────────────────────────────────────

elif page == "📈 Trend Analysis":
    st.title("📈 Claims & Fraud Trend Analysis")

    trend_df = load_monthly_trend()
    trend_df["month"] = pd.to_datetime(trend_df["month"])
    trend_df["fraud_rate"] = trend_df["fraud_claims"] / trend_df["total_claims"].replace(0, 1) * 100

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Monthly Claims Volume")
        chart = (
            alt.Chart(trend_df)
            .mark_area(
                line={"color": "#6366f1"},
                color=alt.Gradient(
                    gradient="linear",
                    stops=[
                        alt.GradientStop(color="#6366f1", offset=0),
                        alt.GradientStop(color="white",   offset=1),
                    ],
                    x1=1, x2=1, y1=1, y2=0,
                ),
            )
            .encode(
                x=alt.X("month:T", title="Month"),
                y=alt.Y("total_claims:Q", title="Claims Count"),
                tooltip=["month:T", "total_claims", "billed_millions"],
            )
            .properties(height=300)
        )
        st.altair_chart(chart, use_container_width=True)

    with col2:
        st.subheader("Monthly Fraud Rate (%)")
        chart = (
            alt.Chart(trend_df)
            .mark_line(point=True, color="#ef4444")
            .encode(
                x=alt.X("month:T", title="Month"),
                y=alt.Y("fraud_rate:Q", title="Fraud Rate (%)"),
                tooltip=["month:T", "fraud_rate", "fraud_claims"],
            )
            .properties(height=300)
        )
        st.altair_chart(chart, use_container_width=True)

    st.subheader("Raw Data")
    st.dataframe(
        trend_df.style.format({
            "billed_millions": "${:.2f}M",
            "fraud_rate":      "{:.2f}%",
        }),
        use_container_width=True,
    )
