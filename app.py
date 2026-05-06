import streamlit as st
import pickle
import pandas as pd
import numpy as np

# --- PAGE CONFIG (must be first) ---
st.set_page_config(
    page_title="HealthShield | Premium Predictor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- ASSET LOADING ---
@st.cache_resource
def load_assets():
    with open("gboost_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("preprocessor.pkl", "rb") as f:
        preprocessor = pickle.load(f)
    return model, preprocessor

model, preprocessor = load_assets()

# --- GLOBAL CSS ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Poppins:wght@600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    min-height: 100vh;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; }

/* Hero Banner */
.hero {
    background: linear-gradient(135deg, rgba(99,102,241,0.25), rgba(168,85,247,0.25));
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    backdrop-filter: blur(12px);
    text-align: center;
}
.hero h1 {
    font-family: 'Poppins', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.5rem 0;
}
.hero p {
    color: rgba(255,255,255,0.65) !important;
    font-size: 1.05rem;
    margin: 0;
}

/* Glass Cards */
.glass-card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 16px;
    padding: 1.8rem;
    backdrop-filter: blur(10px);
    margin-bottom: 1.2rem;
}
.card-title {
    font-family: 'Poppins', sans-serif;
    font-size: 1rem;
    font-weight: 600;
    color: #a78bfa !important;
    -webkit-text-fill-color: #a78bfa !important;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 1.2rem;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* Labels & Inputs */
label, .stSelectbox label, .stNumberInput label {
    color: rgba(255,255,255,0.85) !important;
    -webkit-text-fill-color: rgba(255,255,255,0.85) !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
}
.stSelectbox > div > div,
.stNumberInput > div > div > input {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 10px !important;
    color: white !important;
}

/* Checkboxes */
.stCheckbox {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 10px !important;
    padding: 0.55rem 0.9rem !important;
    margin-bottom: 0.4rem !important;
    transition: background 0.2s;
}
.stCheckbox:hover {
    background: rgba(167,139,250,0.15) !important;
    border-color: rgba(167,139,250,0.5) !important;
}
.stCheckbox label,
.stCheckbox label p,
.stCheckbox label span,
[data-testid="stCheckbox"] label,
[data-testid="stCheckbox"] label p,
[data-testid="stCheckbox"] span,
[data-testid="stCheckbox"] p {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
}
.stCheckbox input[type="checkbox"] {
    accent-color: #a78bfa !important;
    width: 16px !important;
    height: 16px !important;
}

/* Predict Button */
div.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white !important;
    border: none;
    border-radius: 12px;
    padding: 0.85rem 2rem;
    font-size: 1.05rem;
    font-weight: 700;
    font-family: 'Poppins', sans-serif;
    letter-spacing: 0.5px;
    cursor: pointer;
    transition: all 0.3s ease;
    box-shadow: 0 4px 20px rgba(99,102,241,0.4);
    margin-top: 0.5rem;
}
div.stButton > button:hover {
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    box-shadow: 0 6px 28px rgba(99,102,241,0.6);
    transform: translateY(-1px);
}

/* Result Cards */
.result-card {
    border-radius: 18px;
    padding: 2rem 2.5rem;
    text-align: center;
    margin-top: 1rem;
    animation: fadeIn 0.5s ease;
}
.result-low  { background: linear-gradient(135deg, rgba(16,185,129,0.2), rgba(5,150,105,0.2)); border: 1px solid rgba(16,185,129,0.4); }
.result-mid  { background: linear-gradient(135deg, rgba(245,158,11,0.2), rgba(217,119,6,0.2));  border: 1px solid rgba(245,158,11,0.4); }
.result-high { background: linear-gradient(135deg, rgba(239,68,68,0.2),  rgba(185,28,28,0.2));  border: 1px solid rgba(239,68,68,0.4); }

.result-label  { font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 2px; color: rgba(255,255,255,0.6) !important; -webkit-text-fill-color: rgba(255,255,255,0.6) !important; margin-bottom: 0.4rem; }
.result-amount { font-family: 'Poppins', sans-serif; font-size: 3rem; font-weight: 800; color: white !important; -webkit-text-fill-color: white !important; line-height: 1.1; }
.result-tier   { font-size: 0.95rem; font-weight: 500; margin-top: 0.6rem; color: rgba(255,255,255,0.75) !important; -webkit-text-fill-color: rgba(255,255,255,0.75) !important; }
.risk-badge    { display: inline-block; padding: 0.3rem 1rem; border-radius: 20px; font-size: 0.8rem; font-weight: 700; margin-top: 0.8rem; letter-spacing: 1px; }

/* Stat Pills */
.stat-row  { display: flex; gap: 12px; flex-wrap: wrap; margin-top: 1rem; }
.stat-pill { background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.12); border-radius: 10px; padding: 0.6rem 1rem; flex: 1; min-width: 80px; text-align: center; }
.stat-pill .val { font-family: 'Poppins', sans-serif; font-size: 1.3rem; font-weight: 700; color: #a78bfa !important; -webkit-text-fill-color: #a78bfa !important; }
.stat-pill .lbl { font-size: 0.72rem; color: rgba(255,255,255,0.5) !important; -webkit-text-fill-color: rgba(255,255,255,0.5) !important; text-transform: uppercase; letter-spacing: 0.8px; }

.custom-divider { border: none; border-top: 1px solid rgba(255,255,255,0.08); margin: 1.2rem 0 0.8rem; }

.footer { text-align: center; color: rgba(255,255,255,0.3) !important; -webkit-text-fill-color: rgba(255,255,255,0.3) !important; font-size: 0.78rem; margin-top: 2rem; padding-top: 1rem; border-top: 1px solid rgba(255,255,255,0.06); }

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}
</style>
""", unsafe_allow_html=True)

# ── HERO ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1><span style="-webkit-text-fill-color:initial;">🛡️</span> HealthShield Premium Predictor</h1>
    <p>AI-powered insurance premium estimation using Gradient Boosting · Enter patient profile below to get an instant prediction</p>
</div>
""", unsafe_allow_html=True)

# ── LAYOUT ────────────────────────────────────────────────────────────────────
left, right = st.columns([1.1, 0.9], gap="large")

# ════════════════════════════════════════════════════════════════════════════
# LEFT COLUMN — Input Forms
# ════════════════════════════════════════════════════════════════════════════
with left:

    # — Demographics —
    st.markdown('<div class="glass-card"><div class="card-title">👤 Demographics</div>', unsafe_allow_html=True)
    d1, d2 = st.columns(2)
    with d1:
        age = st.number_input("Age", min_value=1, max_value=120, value=35)
    with d2:
        sex = st.selectbox("Sex", ["Male", "Female", "Other"])

    h1, h2, h3 = st.columns(3)
    with h1:
        height_cm = st.number_input("Height (cm)", min_value=50.0, max_value=250.0, value=170.0, step=0.5, format="%.1f")
    with h2:
        weight_kg = st.number_input("Weight (kg)", min_value=10.0, max_value=300.0, value=70.0, step=0.5, format="%.1f")
    with h3:
        # Auto-calculate BMI, but still allow manual override
        auto_bmi = round(weight_kg / ((height_cm / 100) ** 2), 1)
        bmi = st.number_input("BMI (kg/m²) — auto-calculated", min_value=10.0, max_value=60.0,
                              value=float(min(max(auto_bmi, 10.0), 60.0)), step=0.1, format="%.1f")

    s1, s2 = st.columns(2)
    with s1:
        smoker = st.selectbox("Smoking Status", ["Never", "Former", "Current"])
    with s2:
        alcohol = st.selectbox("Alcohol Frequency", ["Never", "Weekly", "Occasional", "Daily"])
    st.markdown('</div>', unsafe_allow_html=True)

    # — Medical History —
    st.markdown('<div class="glass-card"><div class="card-title">🏥 Medical History</div>', unsafe_allow_html=True)
    m1, m2 = st.columns(2)
    with m1:
        med_cost = st.number_input("Annual Medical Cost ($)", min_value=0, max_value=500000, value=3000, step=100)
    with m2:
        hosp = st.selectbox("Hospitalizations (last 3 yrs)", [0, 1], format_func=lambda x: "Yes" if x else "No")
    proc = st.selectbox("Had Major Surgical Procedure", [0, 1], format_func=lambda x: "Yes" if x else "No")
    st.markdown('</div>', unsafe_allow_html=True)

    # — Chronic Conditions (only the 3 the model uses) —
    st.markdown("""
    <div class="glass-card">
        <div class="card-title">🩺 Chronic Conditions</div>
        <p style="color:rgba(255,255,255,0.5);-webkit-text-fill-color:rgba(255,255,255,0.5);font-size:0.78rem;margin:-0.4rem 0 1rem 0;">
            Select all that apply
        </p>
    """, unsafe_allow_html=True)

    ch1, ch2, ch3 = st.columns(3)
    with ch1:
        hyp    = 1 if st.checkbox("◆ Hypertension") else 0
    with ch2:
        dia    = 1 if st.checkbox("◆ Diabetes") else 0
    with ch3:
        arth   = 1 if st.checkbox("◆ Arthritis") else 0

    mental = 1 if st.checkbox("◆ Mental Health Condition") else 0
    st.markdown('</div>', unsafe_allow_html=True)

    predict_btn = st.button("⚡  Predict My Premium")

# ════════════════════════════════════════════════════════════════════════════
# RIGHT COLUMN — Results & Info
# ════════════════════════════════════════════════════════════════════════════
with right:

    # — How it works —
    st.markdown("""
    <div class="glass-card">
        <div class="card-title">⚙️ How It Works</div>
        <p style="color:rgba(255,255,255,0.65);-webkit-text-fill-color:rgba(255,255,255,0.65);font-size:0.88rem;line-height:1.7;margin:0;">
            This tool uses a <strong style="color:#a78bfa;-webkit-text-fill-color:#a78bfa;">Gradient Boosting Regressor</strong>
            trained on real-world health insurance data. It analyses
            <strong style="color:#60a5fa;-webkit-text-fill-color:#60a5fa;">12 clinical &amp; demographic features</strong>
            to estimate your annual premium with high accuracy.<br><br>
            Fill in the form on the left and click <em>Predict</em> to get an instant AI-driven estimate.
        </p>
        <div class="stat-row">
            <div class="stat-pill"><div class="val">GBM</div><div class="lbl">Algorithm</div></div>
            <div class="stat-pill"><div class="val">12</div><div class="lbl">Features</div></div>
            <div class="stat-pill"><div class="val">R²</div><div class="lbl">Optimised</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # — Live Risk Summary —
    chronic_total = hyp + dia + arth + mental
    risk_score = (chronic_total
                  + (2 if smoker == "Current" else 1 if smoker == "Former" else 0)
                  + hosp
                  + (1 if proc else 0))

    if risk_score <= 2:
        risk_label, risk_color = "Low Risk",      "#10b981"
    elif risk_score <= 5:
        risk_label, risk_color = "Moderate Risk", "#f59e0b"
    else:
        risk_label, risk_color = "High Risk",     "#ef4444"

    st.markdown(f"""
    <div class="glass-card">
        <div class="card-title">📊 Live Risk Summary</div>
        <div class="stat-row">
            <div class="stat-pill">
                <div class="val" style="color:{risk_color} !important;-webkit-text-fill-color:{risk_color} !important;">{risk_score}</div>
                <div class="lbl">Risk Score</div>
            </div>
            <div class="stat-pill">
                <div class="val">{chronic_total}</div>
                <div class="lbl">Conditions</div>
            </div>
            <div class="stat-pill">
                <div class="val">{age}</div>
                <div class="lbl">Age</div>
            </div>
            <div class="stat-pill">
                <div class="val">{bmi:.1f}</div>
                <div class="lbl">BMI</div>
            </div>
        </div>
        <div style="margin-top:1rem;">
            <span class="risk-badge" style="background:rgba(255,255,255,0.08);color:{risk_color} !important;-webkit-text-fill-color:{risk_color} !important;border:1px solid {risk_color};">
                ● {risk_label}
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # — Prediction Result —
    if predict_btn:
        try:
            input_data = pd.DataFrame([{
                'age':                      age,
                'sex':                      sex,
                'bmi':                      bmi,
                'smoker':                   smoker,
                'alcohol_freq':             alcohol,
                'hospitalizations_last_3yrs': hosp,
                'annual_medical_cost':      med_cost,
                'hypertension':             hyp,
                'diabetes':                 dia,
                'arthritis':                arth,
                'mental_health':            mental,
                'had_major_procedure':      proc
            }])

            transformed = preprocessor.transform(input_data)
            prediction  = model.predict(transformed)[0]

            if prediction < 800:
                tier, tier_color, card_cls, badge_bg = "Standard Plan", "#10b981", "result-low",  "rgba(16,185,129,0.2)"
            elif prediction < 1500:
                tier, tier_color, card_cls, badge_bg = "Enhanced Plan", "#f59e0b", "result-mid",  "rgba(245,158,11,0.2)"
            else:
                tier, tier_color, card_cls, badge_bg = "Premium Plan",  "#ef4444", "result-high", "rgba(239,68,68,0.2)"

            monthly = prediction / 12

            st.markdown(f"""
            <div class="result-card {card_cls}">
                <div class="result-label">Estimated Annual Premium</div>
                <div class="result-amount">${prediction:,.2f}</div>
                <div class="result-tier">≈ <strong>${monthly:,.2f}</strong> / month</div>
                <span class="risk-badge" style="background:{badge_bg};color:{tier_color} !important;-webkit-text-fill-color:{tier_color} !important;border:1px solid {tier_color};">
                    {tier}
                </span>
                <hr class="custom-divider">
                <p style="font-size:0.82rem;color:#ffffff !important;-webkit-text-fill-color:#ffffff !important;margin:0;background:rgba(255,255,255,0.1);border:1px solid rgba(255,255,255,0.25);border-radius:8px;padding:0.5rem 0.8rem;">
                    ⚠️ Predictive estimate only — not a binding insurance quote.
                </p>
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    🛡️ HealthShield Premium Predictor &nbsp;·&nbsp; Powered by Gradient Boosting &nbsp;·&nbsp; © 2026
</div>
""", unsafe_allow_html=True)
