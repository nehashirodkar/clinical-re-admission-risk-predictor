"""Streamlit dashboard - calls the FastAPI /predict endpoint over HTTP.

Run (with the API already up on :8000):
    streamlit run dashboard/app.py
"""
import os

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Readmission Risk", layout="centered")
st.title("Clinical 30-Day Re-Admission Risk")

DIAG_CATS = [
    "Circulatory", "Respiratory", "Digestive", "Diabetes", "Injury",
    "Musculoskeletal", "Genitourinary", "Neoplasms", "Other", "Missing",
]
AGE_BANDS = {
    "[40-50)": 45, "[50-60)": 55, "[60-70)": 65,
    "[70-80)": 75, "[80-90)": 85, "[90-100)": 95,
}

with st.form("patient"):
    c1, c2 = st.columns(2)
    with c1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        age = st.selectbox("Age band", list(AGE_BANDS))
        time_in_hospital = st.slider("Days in hospital", 1, 14, 4)
        num_medications = st.number_input("Num medications", 0, 100, 15)
        number_inpatient = st.number_input("Prior inpatient visits", 0, 50, 0)
        number_emergency = st.number_input("Prior emergency visits", 0, 50, 0)
        number_outpatient = st.number_input("Prior outpatient visits", 0, 50, 0)
    with c2:
        number_diagnoses = st.number_input("Num diagnoses", 1, 16, 7)
        num_lab_procedures = st.number_input("Lab procedures", 0, 150, 40)
        num_procedures = st.number_input("Procedures", 0, 10, 1)
        diag_1 = st.selectbox("Primary diagnosis", DIAG_CATS)
        diag_2 = st.selectbox("Secondary diagnosis", DIAG_CATS)
        diag_3 = st.selectbox("Tertiary diagnosis", DIAG_CATS)
        n_med_changes = st.number_input("Medication changes", 0, 23, 1)

    submitted = st.form_submit_button("Predict risk")

if submitted:
    total_prior = number_inpatient + number_emergency + number_outpatient
    payload = {
        "race": "Caucasian",
        "gender": gender,
        "age": age,
        "admission_type_id": 1,
        "discharge_disposition_id": 1,
        "admission_source_id": 7,
        "time_in_hospital": time_in_hospital,
        "num_lab_procedures": num_lab_procedures,
        "num_procedures": num_procedures,
        "num_medications": num_medications,
        "number_outpatient": number_outpatient,
        "number_emergency": number_emergency,
        "number_inpatient": number_inpatient,
        "number_diagnoses": number_diagnoses,
        "max_glu_serum": "None",
        "A1Cresult": "None",
        "diag_1_cat": diag_1,
        "diag_2_cat": diag_2,
        "diag_3_cat": diag_3,
        "n_meds_on": max(n_med_changes, 1),
        "n_med_changes": n_med_changes,
        "any_med_change": int(n_med_changes > 0),
        "total_prior_visits": total_prior,
        "had_prior_inpatient": int(number_inpatient > 0),
        "had_prior_emergency": int(number_emergency > 0),
        "age_mid": AGE_BANDS[age],
        "change": "Ch" if n_med_changes > 0 else "No",
        "diabetesMed": "Yes",
        "medical_specialty": "Missing",
    }
    try:
        r = requests.post(f"{API_URL}/predict", json=payload, timeout=30)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        st.error(f"API call failed: {e}")
        st.stop()

    score = data["risk_score"]
    st.metric("Readmission risk", f"{score:.1%}")
    if data["prediction"] == 1:
        st.error(f"HIGH RISK (>= threshold {data['threshold']:.2f})")
    else:
        st.success(f"Lower risk (< threshold {data['threshold']:.2f})")

    st.subheader("Top 5 risk drivers")
    for d in data["top_5_drivers"]:
        arrow = "↑" if d["shap_value"] > 0 else "↓"
        st.write(f"{arrow} **{d['feature']}** — {d['direction']} "
                 f"(SHAP {d['shap_value']:+.3f})")
