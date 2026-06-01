from __future__ import annotations

import hashlib
import json
import random
import re
import secrets
import smtplib
import time
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st
from sklearn.ensemble import RandomForestClassifier


# -------------------------------------------------
# EMAIL OTP SETTINGS
# -------------------------------------------------
def get_secret_value(key: str, default: str = ""):
    try:
        return st.secrets[key]
    except Exception:
        return default


EMAIL_SENDER = get_secret_value("EMAIL_SENDER", "")
EMAIL_PASSWORD = get_secret_value("EMAIL_PASSWORD", "")
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587


# -------------------------------------------------
# PATHS
# -------------------------------------------------
FILE_DIR = Path(__file__).resolve().parent
ROOT_DIR = FILE_DIR.parent

DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"

PROFILE_DB = DATA_DIR / "user_profiles.csv"
LOG_FILE = DATA_DIR / "login_logs.csv"
TRAINING_DATASET = DATA_DIR / "behavior_training_data.csv"

MODEL_FILE = MODELS_DIR / "behavior_model.pkl"
META_FILE = MODELS_DIR / "model_meta.json"
LSTM_META_FILE = MODELS_DIR / "lstm_model_meta.json"

DEFAULT_LOCATIONS = ["Australia", "Nepal", "USA", "India", "Russia", "Other"]

DEFAULT_FEATURES = [
    "avg_typing_speed",
    "location_encoded",
    "login_hour",
    "failed_attempts",
    "location_mismatch",
    "password_length",
    "day_of_week",
]

st.set_page_config(page_title="Fake Login Detection", layout="wide")
st.title("ML-Based Fake Login Detection System")


# -------------------------------------------------
# PASSWORD SECURITY
# -------------------------------------------------
def hash_password(password: str, salt: str) -> str:
    return hashlib.sha256((salt + password).encode()).hexdigest()


def make_salt():
    return secrets.token_hex(16)


def verify_password(input_password: str, salt, stored_hash) -> bool:
    if pd.isna(salt) or pd.isna(stored_hash):
        return False

    salt = str(salt)
    stored_hash = str(stored_hash)

    if salt == "" or stored_hash == "":
        return False

    return hash_password(input_password, salt) == stored_hash


def validate_strong_password(password: str):
    if len(password) < 10:
        return False, "Password must be at least 10 characters."
    if not re.search(r"[A-Z]", password):
        return False, "Add an uppercase letter."
    if not re.search(r"[a-z]", password):
        return False, "Add a lowercase letter."
    if not re.search(r"\d", password):
        return False, "Add a number."
    if not re.search(r"[!@#$%^&*]", password):
        return False, "Add a special symbol."
    return True, "Strong password"


# -------------------------------------------------
# EMAIL OTP
# -------------------------------------------------
def send_email_otp(receiver_email, otp):
    if not EMAIL_SENDER or not EMAIL_PASSWORD:
        st.error("Email secrets are missing. Add EMAIL_SENDER and EMAIL_PASSWORD in Streamlit secrets.")
        return False

    msg = EmailMessage()
    msg["Subject"] = "Your Login OTP Code"
    msg["From"] = EMAIL_SENDER
    msg["To"] = receiver_email

    msg.set_content(
        f"""
Your OTP code is: {otp}

This code is required to complete your login.
Do not share this code with anyone.
"""
    )

    try:
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.send_message(msg)
        server.quit()
        return True
    except Exception as e:
        st.error(f"Email OTP failed: {e}")
        return False


# -------------------------------------------------
# COUNTRY HANDLING
# -------------------------------------------------
def normalize_country(country: str) -> str:
    c = (country or "").strip().lower()

    mapping = {
        "aus": "Australia",
        "australia": "Australia",
        "np": "Nepal",
        "nepal": "Nepal",
        "us": "USA",
        "usa": "USA",
        "united states": "USA",
        "united states of america": "USA",
        "india": "India",
        "russia": "Russia",
        "other": "Other",
        "unknown": "Unknown",
    }

    return mapping.get(c, country.title() if country else "Unknown")


def map_country_to_bucket(country: str) -> str:
    country = normalize_country(country)
    return country if country in DEFAULT_LOCATIONS else "Other"


def bucket_to_code(bucket: str) -> int:
    if bucket not in DEFAULT_LOCATIONS:
        bucket = "Other"
    return DEFAULT_LOCATIONS.index(bucket)


# -------------------------------------------------
# MODEL LOADING
# -------------------------------------------------
@st.cache_resource
def load_meta():
    try:
        if META_FILE.exists():
            return json.loads(META_FILE.read_text())
    except Exception:
        pass

    return {
        "features": DEFAULT_FEATURES,
        "threshold": 0.5,
    }


meta = load_meta()
MODEL_FEATURES = meta.get("features", DEFAULT_FEATURES)
BEST_THR = float(meta.get("threshold", meta.get("threshold_f1_optimal", 0.5)))


@st.cache_resource
def train_rf_from_csv():
    if not TRAINING_DATASET.exists():
        return None

    df = pd.read_csv(TRAINING_DATASET)

    required_cols = DEFAULT_FEATURES + ["label"]
    for col in required_cols:
        if col not in df.columns:
            return None

    df = df.dropna(subset=required_cols)

    X = df[DEFAULT_FEATURES]
    y = df["label"].astype(int)

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
    )

    model.fit(X, y)
    return model


@st.cache_resource
def load_rf_model():
    try:
        if MODEL_FILE.exists():
            return joblib.load(MODEL_FILE)
    except Exception as e:
        st.warning("Saved Random Forest model could not be loaded. A cloud-compatible model will be trained from CSV.")

    try:
        return train_rf_from_csv()
    except Exception as e:
        st.warning(f"Random Forest fallback training failed: {e}")
        return None


@st.cache_resource
def load_lstm_meta():
    try:
        if LSTM_META_FILE.exists():
            return json.loads(LSTM_META_FILE.read_text())
    except Exception:
        pass
    return {}


model = load_rf_model()
lstm_meta = load_lstm_meta()

LSTM_AVAILABLE_IN_CLOUD = False


# -------------------------------------------------
# AUTO LOCATION DETECTION
# -------------------------------------------------
def get_client_ip():
    try:
        return requests.get("https://api.ipify.org?format=json", timeout=5).json()["ip"]
    except Exception:
        return ""


def detect_location():
    ip = get_client_ip()

    try:
        response = requests.get(f"http://ip-api.com/json/{ip}", timeout=5)
        data = response.json()

        if data.get("status") == "success":
            return {
                "ip": ip,
                "country": data.get("country", "Unknown"),
                "region": data.get("regionName", ""),
                "city": data.get("city", ""),
                "zip": data.get("zip", ""),
                "lat": data.get("lat", ""),
                "lon": data.get("lon", ""),
                "isp": data.get("isp", ""),
                "timezone": data.get("timezone", ""),
            }
    except Exception:
        pass

    return {
        "ip": ip,
        "country": "Unknown",
        "region": "",
        "city": "",
        "zip": "",
        "lat": "",
        "lon": "",
        "isp": "",
        "timezone": "",
    }


# -------------------------------------------------
# DATA
# -------------------------------------------------
def load_profiles():
    if PROFILE_DB.exists():
        df = pd.read_csv(PROFILE_DB)
    else:
        df = pd.DataFrame(
            columns=[
                "username",
                "email",
                "password_salt",
                "password_hash",
                "home_country",
                "location",
            ]
        )

    required_columns = [
        "username",
        "email",
        "password_salt",
        "password_hash",
        "home_country",
        "location",
    ]

    for col in required_columns:
        if col not in df.columns:
            df[col] = ""

    df["username"] = df["username"].fillna("").astype(str).str.lower()
    df["email"] = df["email"].fillna("").astype(str).str.lower()
    df["password_salt"] = df["password_salt"].fillna("").astype(str)
    df["password_hash"] = df["password_hash"].fillna("").astype(str)
    df["home_country"] = df["home_country"].fillna("Unknown").astype(str)
    df["location"] = df["location"].fillna("Other").astype(str)

    return df


def save_profiles(df):
    DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(PROFILE_DB, index=False)


def load_logs():
    if LOG_FILE.exists():
        return pd.read_csv(LOG_FILE)
    return pd.DataFrame()


def append_log(row):
    DATA_DIR.mkdir(exist_ok=True)

    if LOG_FILE.exists():
        df = pd.read_csv(LOG_FILE)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df.to_csv(LOG_FILE, index=False)


def count_failed_attempts(username: str) -> int:
    logs = load_logs()

    if logs.empty:
        return 0

    if "username" not in logs.columns or "result" not in logs.columns:
        return 0

    logs["username"] = logs["username"].fillna("").astype(str).str.lower()
    logs["result"] = logs["result"].fillna("").astype(str)

    user_logs = logs[logs["username"] == username.lower()].tail(5)

    return int((user_logs["result"] == "Failed").sum() + (user_logs["result"] == "High Risk").sum())


# -------------------------------------------------
# FEATURE ENGINEERING
# -------------------------------------------------
def build_feature_row(
    speed,
    loc_code,
    login_hour,
    failed_attempts,
    location_mismatch,
    password_length,
    day_of_week,
):
    row = {
        "avg_typing_speed": float(speed),
        "location_encoded": int(loc_code),
        "login_hour": int(login_hour),
        "failed_attempts": int(failed_attempts),
        "location_mismatch": int(location_mismatch),
        "password_length": int(password_length),
        "day_of_week": int(day_of_week),
    }

    for feature in MODEL_FEATURES:
        row.setdefault(feature, 0)

    return pd.DataFrame([row])[MODEL_FEATURES]


# -------------------------------------------------
# RISK SCORING
# -------------------------------------------------
def calculate_risk(prob, typing_speed, failed_attempts, location_mismatch):
    risk_score = 0
    reasons = []

    if prob >= 0.75:
        risk_score += 60
        reasons.append("The machine learning model detected highly unusual behaviour.")
    elif prob >= BEST_THR:
        risk_score += 40
        reasons.append("The machine learning model detected suspicious behaviour.")

    if typing_speed < 0.3:
        risk_score += 20
        reasons.append("Typing speed is very slow.")
    elif typing_speed > 15.0:
        risk_score += 20
        reasons.append("Typing speed is unusually fast.")

    if failed_attempts >= 3:
        risk_score += 25
        reasons.append("Multiple recent failed password attempts were detected.")
    elif failed_attempts >= 1:
        risk_score += 10
        reasons.append("Recent failed password attempts were detected.")

    if location_mismatch == 1:
        risk_score += 25
        reasons.append("The login country is different from the user's home country.")

    return risk_score, reasons


# -------------------------------------------------
# SESSION STATE
# -------------------------------------------------
if "otp_sent" not in st.session_state:
    st.session_state.otp_sent = False

if "generated_otp" not in st.session_state:
    st.session_state.generated_otp = None

if "login_user" not in st.session_state:
    st.session_state.login_user = None

if "login_password" not in st.session_state:
    st.session_state.login_password = None

if "otp_time" not in st.session_state:
    st.session_state.otp_time = None

if "password_typing_start" not in st.session_state:
    st.session_state.password_typing_start = None

if "previous_password_value" not in st.session_state:
    st.session_state.previous_password_value = ""

if "saved_typing_speed" not in st.session_state:
    st.session_state.saved_typing_speed = 0.0

if "login_success" not in st.session_state:
    st.session_state.login_success = False

if "current_location" not in st.session_state:
    st.session_state.current_location = {}


# -------------------------------------------------
# SUCCESS LOCATION PAGE
# -------------------------------------------------
if st.session_state.login_success:
    loc = st.session_state.current_location

    st.title("You are here")
    st.subheader("Your Current Location")

    st.write(f"Country: {loc.get('country', 'Unknown')}")
    st.write(f"Region: {loc.get('region', 'Unknown')}")
    st.write(f"City: {loc.get('city', 'Unknown')}")
    st.write(f"ZIP: {loc.get('zip', 'Unknown')}")
    st.write(f"IP Address: {loc.get('ip', 'Unknown')}")
    st.write(f"ISP: {loc.get('isp', 'Unknown')}")
    st.write(f"Timezone: {loc.get('timezone', 'Unknown')}")
    st.write(f"Coordinates: {loc.get('lat', 'Unknown')}, {loc.get('lon', 'Unknown')}")

    if st.button("Logout"):
        st.session_state.login_success = False
        st.session_state.current_location = {}
        st.rerun()

    st.stop()


# -------------------------------------------------
# UI TABS
# -------------------------------------------------
tab1, tab2 = st.tabs(["Create Account", "Login"])


# -------------------------------------------------
# CREATE ACCOUNT
# -------------------------------------------------
with tab1:
    st.subheader("Create Account")

    username = st.text_input("Username")
    email = st.text_input("Email Address")
    password = st.text_input("Password", type="password")
    confirm = st.text_input("Confirm Password", type="password")

    home_country = st.selectbox(
        "Select Home Country",
        DEFAULT_LOCATIONS,
        help="This is the user's normal home country.",
    )

    if st.button("Create Account"):
        df = load_profiles()

        username_clean = username.strip().lower()
        email_clean = email.strip().lower()

        if not username_clean:
            st.error("Username cannot be empty.")
            st.stop()

        if not email_clean:
            st.error("Email cannot be empty.")
            st.stop()

        if username_clean in df["username"].astype(str).str.lower().values:
            st.error("Username already exists.")
            st.stop()

        valid, msg = validate_strong_password(password)
        if not valid:
            st.error(msg)
            st.stop()

        if password != confirm:
            st.error("Passwords do not match.")
            st.stop()

        salt = make_salt()
        hashed = hash_password(password, salt)

        home_country = normalize_country(home_country)
        home_bucket = map_country_to_bucket(home_country)

        new_user = {
            "username": username_clean,
            "email": email_clean,
            "password_salt": salt,
            "password_hash": hashed,
            "home_country": home_country,
            "location": home_bucket,
        }

        df = pd.concat([df, pd.DataFrame([new_user])], ignore_index=True)
        save_profiles(df)

        st.success("Account created successfully.")


# -------------------------------------------------
# LOGIN
# -------------------------------------------------
with tab2:
    st.subheader("Login")

    username = st.text_input("Username", key="lu")
    password = st.text_input("Password", type="password", key="lp")

    if password and st.session_state.previous_password_value == "":
        st.session_state.password_typing_start = time.time()

    if not password:
        st.session_state.password_typing_start = None
        st.session_state.previous_password_value = ""
    else:
        st.session_state.previous_password_value = password

    st.info(
        "Enter your username and password. Typing speed is measured from password entry until the Login button is clicked."
    )

    if st.button("Login"):
        df = load_profiles()
        username_clean = username.strip().lower()

        user = df[df["username"].astype(str).str.lower() == username_clean]

        if user.empty:
            st.error("User not found.")
            st.stop()

        user = user.iloc[0]

        loc = detect_location()
        login_country = normalize_country(loc.get("country", "Unknown"))

        if st.session_state.password_typing_start is None:
            typing_speed = 2.5
        else:
            duration = max(0.1, time.time() - st.session_state.password_typing_start)
            typing_speed = round(len(password) / duration, 2)

        typing_speed = max(0.3, min(float(typing_speed), 15.0))
        st.session_state.saved_typing_speed = typing_speed

        failed_attempts_now = count_failed_attempts(username_clean) + 1

        if not verify_password(password, user["password_salt"], user["password_hash"]):
            st.error("Wrong password.")

            if failed_attempts_now >= 3:
                failed_result = "High Risk"
                failed_risk_score = 100
                failed_reason = "High alert: 3 or more failed password attempts"
            else:
                failed_result = "Failed"
                failed_risk_score = 70
                failed_reason = "Wrong password attempt"

            append_log(
                {
                    "username": username_clean,
                    "typing_speed": typing_speed,
                    "home_country": user.get("home_country", ""),
                    "login_country": login_country,
                    "login_city": loc.get("city", ""),
                    "login_region": loc.get("region", ""),
                    "login_zip": loc.get("zip", ""),
                    "login_isp": loc.get("isp", ""),
                    "login_timezone": loc.get("timezone", ""),
                    "latitude": loc.get("lat", ""),
                    "longitude": loc.get("lon", ""),
                    "login_ip": loc.get("ip", ""),
                    "login_hour": datetime.now().hour,
                    "failed_attempts": failed_attempts_now,
                    "location_mismatch": 0,
                    "password_length": len(password),
                    "day_of_week": datetime.now().weekday(),
                    "rf_probability": 0,
                    "lstm_probability": 0,
                    "lstm_used": False,
                    "ml_probability": 0,
                    "risk_score": failed_risk_score,
                    "risk_reasons": failed_reason,
                    "result": failed_result,
                    "ts": datetime.now().isoformat(),
                }
            )

            st.session_state.password_typing_start = None
            st.session_state.previous_password_value = ""
            st.stop()

        user_email = str(user.get("email", "")).strip().lower()

        if not user_email or user_email == "nan":
            st.error("No email found for this user. Please create the account with an email address.")
            st.stop()

        otp = str(random.randint(100000, 999999))
        sent = send_email_otp(user_email, otp)

        if sent:
            st.session_state.generated_otp = otp
            st.session_state.otp_sent = True
            st.session_state.login_user = username_clean
            st.session_state.login_password = password
            st.session_state.otp_time = time.time()

            st.success(f"OTP sent to {user_email}.")

    if st.session_state.otp_sent:
        st.markdown("### Enter OTP")

        entered_otp = st.text_input("Enter 6-digit OTP", key="otp_input")

        if st.button("Verify OTP"):
            otp_age = time.time() - st.session_state.otp_time if st.session_state.otp_time else 999

            if otp_age > 300:
                st.error("OTP expired. Please login again.")

                append_log(
                    {
                        "username": st.session_state.login_user,
                        "typing_speed": 0,
                        "home_country": "",
                        "login_country": "Unknown",
                        "rf_probability": 0,
                        "lstm_probability": 0,
                        "lstm_used": False,
                        "ml_probability": 0,
                        "risk_score": 100,
                        "risk_reasons": "OTP expired",
                        "result": "Blocked - OTP Expired",
                        "ts": datetime.now().isoformat(),
                    }
                )

                st.session_state.otp_sent = False
                st.session_state.generated_otp = None
                st.session_state.login_user = None
                st.session_state.login_password = None
                st.session_state.otp_time = None
                st.session_state.password_typing_start = None
                st.session_state.previous_password_value = ""
                st.session_state.saved_typing_speed = 0.0
                st.stop()

            if entered_otp == st.session_state.generated_otp:
                df = load_profiles()
                user = df[
                    df["username"].astype(str).str.lower()
                    == st.session_state.login_user
                ].iloc[0]

                typing_speed = float(st.session_state.saved_typing_speed)
                typing_speed = max(0.3, min(typing_speed, 15.0))

                loc = detect_location()
                login_country = normalize_country(loc.get("country", "Unknown"))
                login_bucket = map_country_to_bucket(login_country)
                loc_code = bucket_to_code(login_bucket)

                home_country = normalize_country(user.get("home_country", "Unknown"))

                location_mismatch = int(
                    home_country != "Unknown"
                    and login_country != "Unknown"
                    and login_country != home_country
                )

                failed_attempts = count_failed_attempts(st.session_state.login_user)

                login_hour = datetime.now().hour
                password_length = len(st.session_state.login_password)
                day_of_week = datetime.now().weekday()

                features = build_feature_row(
                    speed=typing_speed,
                    loc_code=loc_code,
                    login_hour=login_hour,
                    failed_attempts=failed_attempts,
                    location_mismatch=location_mismatch,
                    password_length=password_length,
                    day_of_week=day_of_week,
                )

                try:
                    rf_prob = model.predict_proba(features)[0][1] if model else 0
                except Exception:
                    rf_prob = 0

                lstm_prob = 0
                lstm_used = False

                prob = round(rf_prob, 4)

                risk_score, reasons = calculate_risk(
                    prob=prob,
                    typing_speed=typing_speed,
                    failed_attempts=failed_attempts,
                    location_mismatch=location_mismatch,
                )

                if risk_score < 30:
                    result = "Legitimate"
                elif risk_score < 60:
                    result = "Suspicious"
                else:
                    result = "High Risk"

                append_log(
                    {
                        "username": st.session_state.login_user,
                        "typing_speed": typing_speed,
                        "home_country": home_country,
                        "login_country": login_country,
                        "login_city": loc.get("city", ""),
                        "login_region": loc.get("region", ""),
                        "login_zip": loc.get("zip", ""),
                        "login_isp": loc.get("isp", ""),
                        "login_timezone": loc.get("timezone", ""),
                        "latitude": loc.get("lat", ""),
                        "longitude": loc.get("lon", ""),
                        "login_ip": loc.get("ip", ""),
                        "login_hour": login_hour,
                        "failed_attempts": failed_attempts,
                        "location_mismatch": location_mismatch,
                        "password_length": password_length,
                        "day_of_week": day_of_week,
                        "rf_probability": round(rf_prob, 4),
                        "lstm_probability": round(lstm_prob, 4),
                        "lstm_used": lstm_used,
                        "ml_probability": round(prob, 4),
                        "risk_score": risk_score,
                        "risk_reasons": " | ".join(reasons),
                        "result": result,
                        "ts": datetime.now().isoformat(),
                    }
                )

                st.session_state.login_success = True
                st.session_state.current_location = loc

                st.session_state.otp_sent = False
                st.session_state.generated_otp = None
                st.session_state.login_user = None
                st.session_state.login_password = None
                st.session_state.otp_time = None
                st.session_state.password_typing_start = None
                st.session_state.previous_password_value = ""
                st.session_state.saved_typing_speed = 0.0

                st.rerun()

            else:
                st.error("Wrong OTP.")

                append_log(
                    {
                        "username": st.session_state.login_user,
                        "typing_speed": 0,
                        "home_country": "",
                        "login_country": "Unknown",
                        "rf_probability": 0,
                        "lstm_probability": 0,
                        "lstm_used": False,
                        "ml_probability": 0,
                        "risk_score": 100,
                        "risk_reasons": "Wrong OTP",
                        "result": "Blocked - Wrong OTP",
                        "ts": datetime.now().isoformat(),
                    }
                )
