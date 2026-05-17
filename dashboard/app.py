"""Streamlit dashboard - calls the FastAPI /predict endpoint over HTTP.

Run (with the API already up on :8000):
    streamlit run dashboard/app.py
"""
import os

import plotly.graph_objects as go
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Clinical Readmission Risk",
    page_icon="🏥",
    layout="wide",
)

# ---------------------------------------------------------------- styling
CLINICAL_CSS = """
<style>
:root {
  --c-primary:#1565C0; --c-teal:#0E7C7B; --c-ink:#1A2B3C;
  --c-bg:#F4F7FB; --c-card:#FFFFFF; --c-line:#E2E8F0;
}
.stApp { background: var(--c-bg); }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.4rem; max-width: 1180px; }

.app-header {
  background: linear-gradient(135deg,#1565C0 0%,#0E7C7B 100%);
  color:#fff; padding:1.4rem 1.6rem; border-radius:14px;
  margin-bottom:1.1rem;
}
.app-header h1 { color:#fff; font-size:1.55rem; margin:0; font-weight:700; }
.app-header p  { color:#DCEAF7; margin:.35rem 0 0; font-size:.93rem; }

.section-title {
  font-size:.82rem; font-weight:700; letter-spacing:.06em;
  text-transform:uppercase; color:var(--c-primary);
  margin:.2rem 0 .6rem; border-left:3px solid var(--c-teal);
  padding-left:.55rem;
}
div[data-testid="stForm"] {
  background:var(--c-card); border:1px solid var(--c-line);
  border-radius:14px; padding:1.2rem 1.3rem;
}
.stButton>button, div[data-testid="stFormSubmitButton"] button {
  background:var(--c-primary); color:#fff; border:0;
  border-radius:9px; padding:.55rem 1.4rem; font-weight:600;
  width:100%;
}
.stButton>button:hover, div[data-testid="stFormSubmitButton"] button:hover {
  background:#0E5AA7; color:#fff;
}
.result-card {
  background:var(--c-card); border:1px solid var(--c-line);
  border-radius:14px; padding:1.1rem 1.3rem; height:100%;
}
.risk-band {
  border-radius:10px; padding:.85rem 1rem; font-weight:700;
  font-size:1.02rem; text-align:center; margin-top:.4rem;
}
.band-high { background:#FDECEC; color:#C0392B; border:1px solid #F5B7B1; }
.band-low  { background:#E8F6EF; color:#1E8449; border:1px solid #A9DFBF; }
.driver {
  background:var(--c-card); border:1px solid var(--c-line);
  border-left:5px solid var(--c-line);
  border-radius:9px; padding:.6rem .85rem; margin-bottom:.5rem;
}
.driver.up   { border-left-color:#C0392B; }
.driver.down { border-left-color:#1E8449; }
.driver .feat { font-weight:600; color:var(--c-ink); }
.driver .meta { font-size:.83rem; color:#5A6B7B; }
.disclaimer {
  font-size:.78rem; color:#6B7B8C; margin-top:1.1rem;
  border-top:1px solid var(--c-line); padding-top:.7rem;
}
</style>
"""
st.markdown(CLINICAL_CSS, unsafe_allow_html=True)

st.markdown(
    """
    <div class="app-header">
      <h1>🏥 Clinical 30-Day Re-Admission Risk</h1>
      <p>Decision-support estimate of the probability that a diabetic
      inpatient is re-admitted within 30 days of discharge, with the
      key contributing factors for this patient.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

DIAG_CATS = [
    "Circulatory", "Respiratory", "Digestive", "Diabetes", "Injury",
    "Musculoskeletal", "Genitourinary", "Neoplasms", "Other", "Missing",
]
AGE_BANDS = {
    "[40-50)": 45, "[50-60)": 55, "[60-70)": 65,
    "[70-80)": 75, "[80-90)": 85, "[90-100)": 95,
}


def _clean_feature(name: str) -> str:
    """num__number_inpatient -> 'Number inpatient' (readable for clinicians)."""
    for p in ("num__", "cat__"):
        if name.startswith(p):
            name = name[len(p):]
    return name.replace("_", " ").strip().capitalize()


form_col, result_col = st.columns([1.05, 1], gap="large")

with form_col:
    with st.form("patient"):
        st.markdown('<div class="section-title">Demographics</div>',
                    unsafe_allow_html=True)
        d1, d2 = st.columns(2)
        with d1:
            gender = st.selectbox("Gender", ["Female", "Male"])
        with d2:
            age = st.selectbox("Age band", list(AGE_BANDS))

        st.markdown('<div class="section-title">Current Encounter</div>',
                    unsafe_allow_html=True)
        e1, e2 = st.columns(2)
        with e1:
            time_in_hospital = st.slider("Days in hospital", 1, 14, 4)
            num_lab_procedures = st.number_input("Lab procedures", 0, 150, 40)
            num_procedures = st.number_input("Procedures", 0, 10, 1)
        with e2:
            num_medications = st.number_input("Medications", 0, 100, 15)
            number_diagnoses = st.number_input("Diagnoses", 1, 16, 7)
            n_med_changes = st.number_input("Medication changes", 0, 23, 1)

        st.markdown('<div class="section-title">Prior Utilisation</div>',
                    unsafe_allow_html=True)
        u1, u2, u3 = st.columns(3)
        with u1:
            number_inpatient = st.number_input("Inpatient", 0, 50, 0)
        with u2:
            number_emergency = st.number_input("Emergency", 0, 50, 0)
        with u3:
            number_outpatient = st.number_input("Outpatient", 0, 50, 0)

        st.markdown('<div class="section-title">Diagnoses (ICD-9 group)</div>',
                    unsafe_allow_html=True)
        g1, g2, g3 = st.columns(3)
        with g1:
            diag_1 = st.selectbox("Primary", DIAG_CATS)
        with g2:
            diag_2 = st.selectbox("Secondary", DIAG_CATS)
        with g3:
            diag_3 = st.selectbox("Tertiary", DIAG_CATS)

        submitted = st.form_submit_button("Assess readmission risk")

with result_col:
    if not submitted:
        st.markdown(
            '<div class="result-card"><div class="section-title">'
            'Result</div><p style="color:#5A6B7B;font-size:.92rem;">'
            'Enter the patient encounter details and select '
            '<b>Assess readmission risk</b>. The estimated 30-day '
            'readmission probability and the top contributing factors '
            'will appear here.</p></div>',
            unsafe_allow_html=True,
        )

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
        with result_col:
            st.error(f"API call failed: {e}")
        st.stop()

    score = data["risk_score"]
    threshold = data["threshold"]
    is_high = data["prediction"] == 1

    with result_col:
        st.markdown('<div class="section-title">Estimated Risk</div>',
                    unsafe_allow_html=True)

        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score * 100,
            number={"suffix": "%", "font": {"size": 38,
                                            "color": "#1A2B3C"}},
            gauge={
                "axis": {"range": [0, 50], "tickwidth": 1,
                         "tickcolor": "#94A3B8"},
                "bar": {"color": "#1565C0"},
                "bgcolor": "white",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 8], "color": "#E8F6EF"},
                    {"range": [8, 15], "color": "#FEF5E7"},
                    {"range": [15, 50], "color": "#FDECEC"},
                ],
                "threshold": {
                    "line": {"color": "#C0392B", "width": 3},
                    "thickness": 0.78,
                    "value": threshold * 100,
                },
            },
        ))
        gauge.update_layout(
            height=240, margin=dict(l=20, r=20, t=20, b=10),
            paper_bgcolor="rgba(0,0,0,0)", font={"color": "#1A2B3C"},
        )
        st.plotly_chart(gauge, use_container_width=True,
                         config={"displayModeBar": False})

        band_cls = "band-high" if is_high else "band-low"
        band_txt = (
            f"⚠ HIGH RISK — at or above the {threshold:.0%} decision "
            f"threshold" if is_high else
            f"✓ LOWER RISK — below the {threshold:.0%} decision threshold"
        )
        st.markdown(f'<div class="risk-band {band_cls}">{band_txt}</div>',
                    unsafe_allow_html=True)

        st.markdown('<div class="section-title" style="margin-top:1rem;">'
                    'Top Contributing Factors</div>',
                    unsafe_allow_html=True)
        for d in data["top_5_drivers"]:
            up = d["shap_value"] > 0
            cls = "up" if up else "down"
            arrow = "▲" if up else "▼"
            st.markdown(
                f'<div class="driver {cls}">'
                f'<span class="feat">{arrow} {_clean_feature(d["feature"])}'
                f'</span><br><span class="meta">{d["direction"]} '
                f'· SHAP {d["shap_value"]:+.3f}</span></div>',
                unsafe_allow_html=True,
            )

st.markdown(
    '<div class="disclaimer">⚕ Research / portfolio demonstration only. '
    'Model trained on the UCI Diabetes 130-US Hospitals dataset; test '
    'AUC ≈ 0.64 (a known ceiling for this dataset). Not a medical device '
    'and not for clinical use.</div>',
    unsafe_allow_html=True,
)
