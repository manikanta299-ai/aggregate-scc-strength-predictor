import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="SCC Strength Predictor", layout="wide")

# ------------------------------------------------
# 🔥 PREMIUM UI CSS
# ------------------------------------------------
st.markdown("""
<style>

/* Global font */
html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
}

/* Reduce spacing */
.block-container {
    padding-top: 1rem;
    padding-bottom: 0rem;
}

/* Card layout */
.card {
    background-color: #f8f9fa;
    padding: 15px;
    border-radius: 10px;
    box-shadow: 0px 2px 8px rgba(0,0,0,0.1);
    margin-bottom: 10px;
}

/* Section title */
.section-title {
    font-size: 22px;
    font-weight: bold;
    margin-bottom: 10px;
}

/* Input labels */
label {
    font-size: 17px !important;
    font-weight: 600 !important;
}

/* Inputs */
input, .stNumberInput input {
    font-size: 17px !important;
}

/* Selectbox */
div[data-baseweb="select"] > div {
    font-size: 17px !important;
}

/* KPI box */
.kpi {
    background-color: #e8f5e9;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    font-size: 28px;
    font-weight: bold;
    color: #1b5e20;
}

/* Sticky header */
header {
    position: sticky;
    top: 0;
    background-color: white;
    z-index: 999;
}

/* Button */
.stButton button {
    font-size: 18px;
    padding: 10px 25px;
    border-radius: 8px;
}

</style>
""", unsafe_allow_html=True)

# ------------------------------------------------
# TITLE
# ------------------------------------------------
st.title("AI-Based SCC Compressive Strength Prediction")
st.markdown("### Optimized ELM-CMAES Model")

# ------------------------------------------------
# MATERIAL MAP (NO FCS)
# ------------------------------------------------
material_map = {
    0: "Blast Furnace Slag (BFS)",
    1: "Coal Bottom Ash (CBA)",
    2: "Copper Slag (CS)",
    3: "Ceramic Waste (CW)",
    5: "Iron Slag (IS)",
    6: "Steel Slag (SS)",
    7: "Foundry Sand (FS)"
}

material_names = list(material_map.values())
material_reverse = {v: k for k, v in material_map.items()}

# ------------------------------------------------
# LOAD MODEL
# ------------------------------------------------
model = joblib.load("ELM_CMAES.pkl")

# ------------------------------------------------
# ELM FUNCTION
# ------------------------------------------------
def elm_predict(X_df, model_dict):
    W = model_dict["W"]
    b = model_dict["b"]
    beta = model_dict["beta"]
    scaler = model_dict["scaler"]

    X = np.array(X_df).reshape(1, -1)
    X_scaled = scaler.transform(X)
    H = 1 / (1 + np.exp(-(X_scaled @ W + b)))

    return H @ beta

# ------------------------------------------------
# INPUT SECTION (CARD STYLE)
# ------------------------------------------------
st.markdown('<div class="section-title">🔧 Input Parameters</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    material = st.selectbox("Material Type", material_names)
    replacement = st.slider("% Replacement", 0.0, 100.0, 10.0)
    binder = st.number_input("Binder (kg/m³)", value=400.0)
    wb = st.number_input("Water/Binder Ratio", value=0.55)
    fine_agg = st.number_input("Fine Aggregate", value=829.6)
    coarse_agg = st.number_input("Coarse Aggregate", value=656.0)

with col2:
    sp = st.number_input("Superplasticizer (%)", value=1.05)
    sio2 = st.number_input("SiO2 (%)", value=30.2)
    cao = st.number_input("CaO (%)", value=39.5)
    al2o3 = st.number_input("Al2O3 (%)", value=3.0)
    fe2o3 = st.number_input("Fe2O3 (%)", value=31.8)
    sg = st.number_input("Material Specific Gravity", value=2.45)

col3, col4 = st.columns(2)

with col3:
    wa = st.number_input("Water Absorption (%)", value=8.0)
    fm = st.number_input("Fineness Modulus", value=3.11)
    slump = st.number_input("Slump (mm)", value=705.0)

with col4:
    t500 = st.number_input("T500 (sec)", value=3.4)
    age = st.number_input("Age (days)", value=7)

material_code = material_reverse[material]

# ------------------------------------------------
# PREDICTION BUTTON
# ------------------------------------------------
if st.button("🚀 Predict Compressive Strength"):

    input_df = pd.DataFrame([[
        material_code, replacement, binder, wb,
        fine_agg, coarse_agg, sp,
        sio2, cao, al2o3, fe2o3,
        sg, wa, fm, slump, t500, age
    ]], columns=[
        "Material_Name", "%Replace", "Binder", "w/b",
        "Fine_Agg", "Coarse_Agg", "SP",
        "SiO2", "CaO", "Al2O3", "Fe2O3",
        "Material_SG", "Material_WA", "Material_FM",
        "Slump", "T50", "Age"
    ])

    prediction = elm_predict(input_df, model)[0]

    # ------------------------------------------------
    # KPI OUTPUT
    # ------------------------------------------------
    st.markdown('<div class="section-title">📊 Prediction Result</div>', unsafe_allow_html=True)

    st.markdown(f'<div class="kpi">{prediction:.2f} MPa</div>', unsafe_allow_html=True)

    # Strength category
    if prediction < 30:
        st.error("Low Strength Mix")
    elif prediction < 60:
        st.warning("Moderate Strength Mix")
    else:
        st.success("High Strength SCC")

    # ------------------------------------------------
    # SENSITIVITY PLOT
    # ------------------------------------------------
    st.subheader("📈 Replacement Sensitivity")

    rep_range = np.linspace(0, 60, 30)
    preds = []

    for r in rep_range:
        temp = input_df.copy()
        temp["%Replace"] = r
        preds.append(elm_predict(temp, model)[0])

    fig1 = plt.figure()
    plt.plot(rep_range, preds)
    plt.xlabel("Replacement (%)")
    plt.ylabel("Strength (MPa)")
    plt.grid()
    st.pyplot(fig1)

    # ------------------------------------------------
    # FEATURE IMPACT
    # ------------------------------------------------
    st.subheader("📊 Feature Influence")

    impacts = []
    for col in input_df.columns:
        temp = input_df.copy()
        temp[col] *= 1.05
        impacts.append(elm_predict(temp, model)[0] - prediction)

    fig2 = plt.figure()
    plt.barh(input_df.columns, impacts)
    plt.grid()
    st.pyplot(fig2)

    # ------------------------------------------------
    # INPUT SUMMARY
    # ------------------------------------------------
    with st.expander("📄 View Input Summary"):
        st.dataframe(input_df)

    # ------------------------------------------------
    # VALIDATION
    # ------------------------------------------------
    st.subheader("🎯 Validation")

    actual = st.number_input("Actual Strength", value=0.0)

    if actual > 0:
        error = abs(prediction - actual)
        st.write(f"Error: {error:.2f} MPa")

        if error < 2:
            st.success("Excellent Prediction")
        elif error < 5:
            st.info("Good Prediction")
        else:
            st.warning("High Deviation")
