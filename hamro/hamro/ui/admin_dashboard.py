from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st


# -------------------------------------------------
# PATHS
# -------------------------------------------------
FILE_DIR = Path(__file__).resolve().parent
ROOT_DIR = FILE_DIR.parent

DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"

LOG_FILE = DATA_DIR / "login_logs.csv"
RF_META_FILE = MODELS_DIR / "model_meta.json"
LSTM_META_FILE = MODELS_DIR / "lstm_model_meta.json"


# -------------------------------------------------
# PAGE SETUP
# -------------------------------------------------
st.set_page_config(page_title="Admin Dashboard", layout="wide")

st.title("Admin Dashboard")
st.caption(
    "Behavioural login monitoring with Random Forest and LSTM-based risk analysis."
)


# -------------------------------------------------
# LOAD METADATA
# -------------------------------------------------
def load_json(path: Path):
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return {}
    return {}


rf_meta = load_json(RF_META_FILE)
lstm_meta = load_json(LSTM_META_FILE)


# -------------------------------------------------
# LOAD LOGS
# -------------------------------------------------
def load_logs():
    if not LOG_FILE.exists():
        return pd.DataFrame()

    df = pd.read_csv(LOG_FILE)

    required_cols = [
        "username",
        "typing_speed",
        "home_country",
        "login_country",
        "login_city",
        "login_region",
        "login_zip",
        "login_isp",
        "login_timezone",
        "latitude",
        "longitude",
        "login_ip",
        "login_hour",
        "failed_attempts",
        "location_mismatch",
        "password_length",
        "day_of_week",
        "rf_probability",
        "lstm_probability",
        "lstm_used",
        "ml_probability",
        "risk_score",
        "risk_reasons",
        "result",
        "ts",
    ]

    for col in required_cols:
        if col not in df.columns:
            df[col] = ""

    text_cols = [
        "username",
        "home_country",
        "login_country",
        "login_city",
        "login_region",
        "login_zip",
        "login_isp",
        "login_timezone",
        "login_ip",
        "risk_reasons",
        "result",
    ]

    for col in text_cols:
        df[col] = df[col].fillna("").astype(str)

    df["username"] = df["username"].str.lower()
    df["result"] = df["result"].replace("", "Unknown")

    numeric_cols = [
        "typing_speed",
        "login_hour",
        "failed_attempts",
        "location_mismatch",
        "password_length",
        "day_of_week",
        "rf_probability",
        "lstm_probability",
        "ml_probability",
        "risk_score",
        "latitude",
        "longitude",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["lstm_used"] = (
        df["lstm_used"]
        .astype(str)
        .str.lower()
        .isin(["true", "1", "yes"])
    )

    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    df["date"] = df["ts"].dt.date
    df["hour"] = df["ts"].dt.hour.fillna(0).astype(int)

    df["calculated_mismatch"] = (
        df["home_country"].str.lower() != df["login_country"].str.lower()
    )

    df.loc[
        df["home_country"].str.lower().isin(["", "unknown", "nan"]),
        "calculated_mismatch",
    ] = False

    df.loc[
        df["login_country"].str.lower().isin(["", "unknown", "nan"]),
        "calculated_mismatch",
    ] = False

    return df


logs = load_logs()

if logs.empty:
    st.warning("No login logs found yet.")
    st.stop()


# -------------------------------------------------
# SIDEBAR FILTERS
# -------------------------------------------------
st.sidebar.header("Filters")

users = sorted(logs["username"].dropna().unique().tolist())
selected_users = st.sidebar.multiselect("Users", users)

results = sorted(logs["result"].dropna().unique().tolist())
selected_results = st.sidebar.multiselect("Login Result", results, default=results)

only_risky = st.sidebar.checkbox("Only Risky Attempts")
only_lstm_used = st.sidebar.checkbox("Only LSTM Used")
only_location_mismatch = st.sidebar.checkbox("Only Location Mismatch")

risk_range = st.sidebar.slider("Risk Score Range", 0, 100, (0, 100))
prob_range = st.sidebar.slider("ML Probability Range", 0.0, 1.0, (0.0, 1.0))

filtered = logs.copy()

if selected_users:
    filtered = filtered[filtered["username"].isin(selected_users)]

if selected_results:
    filtered = filtered[filtered["result"].isin(selected_results)]

if only_risky:
    filtered = filtered[
        filtered["result"].isin(
            [
                "Suspicious",
                "High Risk",
                "Failed",
                "Blocked - Wrong OTP",
                "Blocked - OTP Expired",
                "Blocked - No Email",
            ]
        )
    ]

if only_lstm_used:
    filtered = filtered[filtered["lstm_used"] == True]

if only_location_mismatch:
    filtered = filtered[filtered["calculated_mismatch"] == True]

filtered = filtered[
    (filtered["risk_score"] >= risk_range[0])
    & (filtered["risk_score"] <= risk_range[1])
    & (filtered["ml_probability"] >= prob_range[0])
    & (filtered["ml_probability"] <= prob_range[1])
]


# -------------------------------------------------
# KPI CARDS
# -------------------------------------------------
total = len(filtered)
legitimate = int((filtered["result"] == "Legitimate").sum())
suspicious = int((filtered["result"] == "Suspicious").sum())
high_risk = int((filtered["result"] == "High Risk").sum())
failed = int((filtered["result"] == "Failed").sum())
blocked = int(filtered["result"].str.startswith("Blocked").sum())
lstm_used_count = int(filtered["lstm_used"].sum())

avg_risk = round(filtered["risk_score"].mean(), 2) if total else 0
avg_rf = round(filtered["rf_probability"].mean(), 3) if total else 0
avg_lstm = round(filtered["lstm_probability"].mean(), 3) if total else 0
avg_ml = round(filtered["ml_probability"].mean(), 3) if total else 0
avg_speed = round(filtered["typing_speed"].mean(), 2) if total else 0

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Total Attempts", total)
k2.metric("Legitimate", legitimate)
k3.metric("Suspicious", suspicious)
k4.metric("High Risk", high_risk)
k5.metric("Failed", failed)
k6.metric("Blocked", blocked)

k7, k8, k9, k10, k11 = st.columns(5)
k7.metric("Average Risk Score", avg_risk)
k8.metric("RF Probability", avg_rf)
k9.metric("LSTM Probability", avg_lstm)
k10.metric("Final ML Probability", avg_ml)
k11.metric("LSTM Used", lstm_used_count)

st.metric("Average Typing Speed", f"{avg_speed} chars/sec")


# -------------------------------------------------
# MODEL SUMMARY
# -------------------------------------------------
st.subheader("Model Summary")

rf_col, lstm_col = st.columns(2)

with rf_col:
    st.markdown("#### Random Forest")

    if rf_meta:
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Accuracy", round(rf_meta.get("accuracy", 0), 4))
        m2.metric("Precision", round(rf_meta.get("precision", 0), 4))
        m3.metric("Recall", round(rf_meta.get("recall", 0), 4))
        m4.metric("F1", round(rf_meta.get("f1", 0), 4))
        m5.metric("ROC-AUC", round(rf_meta.get("roc_auc", 0), 4))

        st.write("Features:", ", ".join(rf_meta.get("features", [])))
    else:
        st.info("Random Forest metadata not found.")

with lstm_col:
    st.markdown("#### LSTM")

    if lstm_meta:
        l1, l2, l3, l4, l5 = st.columns(5)
        l1.metric("Accuracy", round(lstm_meta.get("accuracy", 0), 4))
        l2.metric("Precision", round(lstm_meta.get("precision", 0), 4))
        l3.metric("Recall", round(lstm_meta.get("recall", 0), 4))
        l4.metric("F1", round(lstm_meta.get("f1", 0), 4))
        l5.metric("ROC-AUC", round(lstm_meta.get("roc_auc", 0), 4))

        st.write("Sequence Length:", lstm_meta.get("sequence_length", "N/A"))
        st.write("Features:", ", ".join(lstm_meta.get("features", [])))
    else:
        st.info("LSTM metadata not found.")


# -------------------------------------------------
# SECURITY ALERTS
# -------------------------------------------------
st.subheader("Security Alerts")

alert_left, alert_right = st.columns(2)

with alert_left:
    st.markdown("#### Multiple Failed or Blocked Attempts")

    failed_logs = filtered[
        filtered["result"].isin(
            ["Failed","High Risk","Blocked - Wrong OTP", "Blocked - OTP Expired"]
        )
    ]

    if failed_logs.empty:
        st.success("No failed or blocked attempts found.")
    else:
        failed_summary = (
            failed_logs.groupby("username")
            .size()
            .reset_index(name="failed_or_blocked_attempts")
            .sort_values("failed_or_blocked_attempts", ascending=False)
        )

        failed_summary = failed_summary[failed_summary["failed_or_blocked_attempts"] >= 2]

        if failed_summary.empty:
            st.success("No user has 2 or more failed or blocked attempts.")
        else:
            st.dataframe(failed_summary, use_container_width=True)

with alert_right:
    st.markdown("#### Suspicious and High Risk Users")

    risky_users = filtered[filtered["result"].isin(["Suspicious", "High Risk"])]

    if risky_users.empty:
        st.success("No suspicious or high-risk users found.")
    else:
        risky_summary = (
            risky_users.groupby("username")
            .agg(
                attempts=("username", "size"),
                avg_risk=("risk_score", "mean"),
                max_risk=("risk_score", "max"),
                avg_rf_probability=("rf_probability", "mean"),
                avg_lstm_probability=("lstm_probability", "mean"),
                avg_ml_probability=("ml_probability", "mean"),
            )
            .reset_index()
            .sort_values("max_risk", ascending=False)
        )

        for col in [
            "avg_risk",
            "avg_rf_probability",
            "avg_lstm_probability",
            "avg_ml_probability",
        ]:
            risky_summary[col] = risky_summary[col].round(3)

        st.dataframe(risky_summary, use_container_width=True)


# -------------------------------------------------
# LOGIN TRENDS
# -------------------------------------------------
st.subheader("Login Trends")

if filtered["ts"].notna().any():
    trend = (
        filtered.groupby([filtered["ts"].dt.date, "result"])
        .size()
        .unstack(fill_value=0)
    )
    st.area_chart(trend)
else:
    st.info("No valid timestamp data available.")


# -------------------------------------------------
# DISTRIBUTIONS
# -------------------------------------------------
dist_left, dist_right = st.columns(2)

with dist_left:
    st.subheader("Login Result Distribution")
    st.bar_chart(filtered["result"].value_counts())

with dist_right:
    st.subheader("Average Risk by Result")

    if not filtered.empty:
        st.bar_chart(filtered.groupby("result")["risk_score"].mean().sort_values())
    else:
        st.info("No data available.")


# -------------------------------------------------
# MODEL PROBABILITY COMPARISON
# -------------------------------------------------
st.subheader("Model Probability Comparison")

prob_cols = ["rf_probability", "lstm_probability", "ml_probability"]

prob_data = filtered[prob_cols].copy()
prob_data = prob_data.rename(
    columns={
        "rf_probability": "Random Forest",
        "lstm_probability": "LSTM",
        "ml_probability": "Final Combined",
    }
)

st.line_chart(prob_data.reset_index(drop=True))


# -------------------------------------------------
# LOCATION ANALYSIS
# -------------------------------------------------
st.subheader("Location Analysis")

loc_summary = (
    filtered.groupby(["home_country", "login_country", "result"])
    .size()
    .reset_index(name="attempts")
    .sort_values("attempts", ascending=False)
)

st.dataframe(loc_summary, use_container_width=True)

country_summary = (
    filtered.groupby("login_country")
    .agg(
        attempts=("login_country", "size"),
        avg_risk=("risk_score", "mean"),
        suspicious=("result", lambda x: (x == "Suspicious").sum()),
        high_risk=("result", lambda x: (x == "High Risk").sum()),
        failed=("result", lambda x: (x == "Failed").sum()),
        lstm_used=("lstm_used", "sum"),
    )
    .reset_index()
    .sort_values("attempts", ascending=False)
)

country_summary["avg_risk"] = country_summary["avg_risk"].round(2)

st.markdown("#### Country Risk Summary")
st.dataframe(country_summary, use_container_width=True)


# -------------------------------------------------
# USER BEHAVIOUR SUMMARY
# -------------------------------------------------
st.subheader("User Behaviour Summary")

user_summary = (
    filtered.groupby("username")
    .agg(
        total_attempts=("username", "size"),
        legitimate=("result", lambda x: (x == "Legitimate").sum()),
        suspicious=("result", lambda x: (x == "Suspicious").sum()),
        high_risk=("result", lambda x: (x == "High Risk").sum()),
        failed=("result", lambda x: (x == "Failed").sum()),
        blocked=("result", lambda x: x.astype(str).str.startswith("Blocked").sum()),
        avg_typing_speed=("typing_speed", "mean"),
        avg_rf_probability=("rf_probability", "mean"),
        avg_lstm_probability=("lstm_probability", "mean"),
        avg_ml_probability=("ml_probability", "mean"),
        avg_risk_score=("risk_score", "mean"),
        location_mismatch_rate=("calculated_mismatch", "mean"),
        lstm_used_count=("lstm_used", "sum"),
    )
    .reset_index()
)

for col in [
    "avg_typing_speed",
    "avg_rf_probability",
    "avg_lstm_probability",
    "avg_ml_probability",
    "avg_risk_score",
]:
    user_summary[col] = user_summary[col].round(3)

user_summary["location_mismatch_rate"] = (
    user_summary["location_mismatch_rate"] * 100
).round(1)

user_summary = user_summary.sort_values("avg_risk_score", ascending=False)

st.dataframe(user_summary, use_container_width=True)


# -------------------------------------------------
# TYPING SPEED ANALYSIS
# -------------------------------------------------
st.subheader("Typing Speed Analysis")

speed_left, speed_right = st.columns(2)

with speed_left:
    speed_by_result = (
        filtered.groupby("result")["typing_speed"]
        .mean()
        .sort_values(ascending=False)
    )

    st.markdown("#### Average Typing Speed by Result")
    st.bar_chart(speed_by_result)

with speed_right:
    unusual_speed = filtered[
        (filtered["typing_speed"] < 0.3)
        | (filtered["typing_speed"] > 15.0)
    ]

    st.markdown("#### Unusual Typing Speed Records")

    if unusual_speed.empty:
        st.success("No unusual typing speed records found.")
    else:
        st.dataframe(
            unusual_speed[
                [
                    "ts",
                    "username",
                    "typing_speed",
                    "result",
                    "risk_score",
                    "risk_reasons",
                ]
            ].sort_values("ts", ascending=False),
            use_container_width=True,
        )


# -------------------------------------------------
# DETAILED LOGIN LOGS
# -------------------------------------------------
st.subheader("Detailed Login Logs")

display_cols = [
    "ts",
    "username",
    "result",
    "risk_score",
    "rf_probability",
    "lstm_probability",
    "lstm_used",
    "ml_probability",
    "risk_reasons",
    "typing_speed",
    "login_hour",
    "failed_attempts",
    "location_mismatch",
    "password_length",
    "day_of_week",
    "home_country",
    "login_country",
    "login_city",
    "login_region",
    "login_zip",
    "login_isp",
    "login_timezone",
    "login_ip",
    "latitude",
    "longitude",
]

for col in display_cols:
    if col not in filtered.columns:
        filtered[col] = ""

detailed_logs = filtered[display_cols].sort_values("ts", ascending=False)

st.dataframe(detailed_logs, use_container_width=True)


# -------------------------------------------------
# DOWNLOAD
# -------------------------------------------------
csv = detailed_logs.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Filtered Logs as CSV",
    data=csv,
    file_name="filtered_login_logs.csv",
    mime="text/csv",
)