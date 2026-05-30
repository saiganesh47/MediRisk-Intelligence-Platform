import streamlit as st
import pandas as pd
import altair as alt
from pathlib import Path

st.set_page_config(page_title="MediRisk Dashboard", page_icon="🏥", layout="wide")

BASE = Path("data")

@st.cache_data
def load_scores():
    parts = list((BASE / "scored/fraud_scores_snowpipe").glob("*.parquet"))
    df = pd.concat([pd.read_parquet(p) for p in parts])
    for col in df.columns:
        if "date" in col.lower() or "time" in col.lower() or "ts" in col.lower():
            try:
                df[col] = pd.to_datetime(df[col], utc=True).dt.tz_localize(None)
            except Exception:
                pass
    return df

@st.cache_data
def load_claims():
    df = pd.read_parquet(BASE / "synthetic/output/claims.parquet")
    for col in df.columns:
        if "date" in col.lower():
            try:
                df[col] = pd.to_datetime(df[col], utc=True).dt.tz_localize(None)
            except Exception:
                pass
    return df

# ── Sidebar ───────────────────────────────────────────────────────
st.sidebar.title("🏥 MediRisk")
st.sidebar.caption("Fraud & Risk Intelligence Platform")
page = st.sidebar.radio("Navigation", [
    "📊 Overview", "🚨 Fraud Alerts", "📈 Trends"
])

scores = load_scores()

# ── Overview ──────────────────────────────────────────────────────
if page == "📊 Overview":
    st.title("📊 Executive Overview")
    st.caption("Real pipeline output — 100,000 scored insurance claims")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Claims",   f"{len(scores):,}")
    c2.metric("🔴 HIGH Risk",   f"{len(scores[scores['risk_band']=='HIGH']):,}")
    c3.metric("🟡 MEDIUM Risk", f"{len(scores[scores['risk_band']=='MEDIUM']):,}")
    c4.metric("Avg Fraud Prob", f"{scores['fraud_probability'].mean():.2%}")

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Risk Band Distribution")
        dist = scores["risk_band"].value_counts().reset_index()
        dist.columns = ["risk_band", "count"]
        colors = {
            "HIGH":   "#ef4444",
            "MEDIUM": "#f59e0b",
            "LOW":    "#22c55e",
            "CLEAR":  "#6366f1"
        }
        chart = (
            alt.Chart(dist)
            .mark_arc(innerRadius=60)
            .encode(
                theta=alt.Theta("count:Q"),
                color=alt.Color("risk_band:N",
                    scale=alt.Scale(
                        domain=list(colors.keys()),
                        range=list(colors.values())
                    )
                ),
                tooltip=["risk_band", "count"]
            )
            .properties(height=320)
        )
        st.altair_chart(chart, use_container_width=True)

    with col2:
        st.subheader("Fraud Probability Distribution")
        sample = scores.sample(min(5000, len(scores)))
        chart = (
            alt.Chart(sample)
            .mark_bar(color="#6366f1")
            .encode(
                x=alt.X("fraud_probability:Q",
                        bin=alt.Bin(maxbins=30),
                        title="Fraud Probability"),
                y=alt.Y("count():Q", title="Number of Claims"),
                tooltip=["count()"]
            )
            .properties(height=320)
        )
        st.altair_chart(chart, use_container_width=True)

    st.divider()
    st.subheader("Model Performance")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("AUC-ROC",   "0.9997 🔥")
    m2.metric("Accuracy",  "99.89%")
    m3.metric("Precision", "99.89%")
    m4.metric("Recall",    "99.89%")

# ── Fraud Alerts ───────────────────────────────────────────────────
elif page == "🚨 Fraud Alerts":
    st.title("🚨 Fraud Alerts — HIGH Risk Claims")

    high = scores[scores["risk_band"] == "HIGH"].sort_values(
        "fraud_probability", ascending=False
    )

    threshold = st.slider("Min Fraud Probability", 0.5, 1.0, 0.8, 0.01)
    filtered  = high[high["fraud_probability"] >= threshold]

    st.metric("Matching Alerts", f"{len(filtered):,}")

    display_cols = [c for c in [
        "claim_id", "patient_id", "provider_id",
        "service_date", "billed_amount",
        "fraud_probability", "risk_band"
    ] if c in filtered.columns]

    fmt = {}
    if "billed_amount"     in filtered.columns: fmt["billed_amount"]     = "${:,.2f}"
    if "fraud_probability" in filtered.columns: fmt["fraud_probability"] = "{:.2%}"

    st.dataframe(
        filtered[display_cols].style.format(fmt),
        use_container_width=True,
        height=500,
    )

    st.divider()
    st.subheader("Fraud Probability — TOP 50 Claims")
    top50 = filtered.head(50)
    if len(top50) > 0:
        chart = (
            alt.Chart(top50.reset_index())
            .mark_bar(color="#ef4444")
            .encode(
                x=alt.X("fraud_probability:Q", title="Fraud Probability"),
                y=alt.Y("index:O",             title="Claim Index", sort="-x"),
                tooltip=["claim_id", "fraud_probability", "billed_amount"]
                        if "billed_amount" in top50.columns else
                        ["claim_id", "fraud_probability"]
            )
            .properties(height=400)
        )
        st.altair_chart(chart, use_container_width=True)

# ── Trends ─────────────────────────────────────────────────────────
elif page == "📈 Trends":
    st.title("📈 Claims & Fraud Trend Analysis")

    claims = load_claims()
    claims["service_date"] = pd.to_datetime(claims["service_date"], errors="coerce")
    claims["month"] = claims["service_date"].dt.to_period("M").astype(str)

    monthly = (
        claims
        .groupby("month")
        .agg(
            total_claims =("claim_id",      "count"),
            total_billed =("billed_amount", "sum"),
            fraud_count  =("is_fraud",      "sum"),
        )
        .reset_index()
    )
    monthly["fraud_rate"] = (
        monthly["fraud_count"] / monthly["total_claims"] * 100
    ).round(2)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Monthly Claims Volume")
        chart = (
            alt.Chart(monthly)
            .mark_line(point=True, color="#6366f1")
            .encode(
                x=alt.X("month:O",        title="Month"),
                y=alt.Y("total_claims:Q", title="Claims"),
                tooltip=["month", "total_claims"]
            )
            .properties(height=300)
        )
        st.altair_chart(chart, use_container_width=True)

    with col2:
        st.subheader("Monthly Fraud Rate (%)")
        chart = (
            alt.Chart(monthly)
            .mark_line(point=True, color="#ef4444")
            .encode(
                x=alt.X("month:O",      title="Month"),
                y=alt.Y("fraud_rate:Q", title="Fraud Rate (%)"),
                tooltip=["month", "fraud_rate", "fraud_count"]
            )
            .properties(height=300)
        )
        st.altair_chart(chart, use_container_width=True)

    st.subheader("Monthly Summary Table")
    st.dataframe(
        monthly.style.format({
            "total_billed": "${:,.0f}",
            "fraud_rate":   "{:.2f}%",
        }),
        use_container_width=True,
    )