import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="SCC Strength Predictor (ELM)", layout="wide")

st.title("AI-Based SCC Compressive Strength Prediction (ELM Models)")

# ------------------------------------------------
# MATERIAL MAPPING
# ------------------------------------------------

material_map = {
    0: "Blast Furnace Slag",
    1: "Coal Bottom Ash",
    2: "Copper Slag",
    3: "Ceramic Waste",
    4: "Foundry Casting Sand",
    5: "Iron Slag",
    6: "Steel Slag",
    7: "Foundry Sand"
}

material_names = list(material_map.values())
material_reverse = {v: k for k, v in material_map.items()}

# ------------------------------------------------
# LOAD MODELS
# ------------------------------------------------

model_files = {
    "ELM_PSO": "ELM_PSO.pkl",
    "ELM_GWO": "ELM_GWO.pkl",
    "ELM_WOA": "ELM_WOA.pkl",
    "ELM_CMAES": "ELM_CMAES.pkl"
}

models = {}
for name, path in model_files.items():
    models[name] = joblib.load(path)

# ------------------------------------------------
# INPUT SECTION
# ------------------------------------------------

st.subheader("Input Parameters")

col1, col2, col3 = st.columns(3)

with col1:
    material = st.selectbox("Material Type", material_names)
    replacement = st.number_input("% Replace", value=10.0)
    binder = st.number_input("Binder (kg/m³)", value=400.0)

with col2:
    wb = st.number_input("Water/Binder Ratio", value=0.55)
    fine_agg = st.number_input("Fine Aggregate", value=829.6)
    coarse_agg = st.number_input("Coarse Aggregate", value=656.0)

with col3:
    sp = st.number_input("Superplasticizer (%)", value=1.05)
    sio2 = st.number_input("SiO2 (%)", value=30.2)
    cao = st.number_input("CaO (%)", value=39.5)

col4, col5, col6 = st.columns(3)

with col4:
    al2o3 = st.number_input("Al2O3 (%)", value=3.0)
    fe2o3 = st.number_input("Fe2O3 (%)", value=31.8)

with col5:
    sg = st.number_input("Material Specific Gravity", value=2.45)
    wa = st.number_input("Water Absorption (%)", value=8.0)

with col6:
    fm = st.number_input("Fineness Modulus", value=3.11)
    slump = st.number_input("Slump (mm)", value=705.0)

col7, col8 = st.columns(2)

with col7:
    t500 = st.number_input("T500 (sec)", value=3.4)

with col8:
    age = st.number_input("Age (days)", value=7)

material_code = material_reverse[material]

# ------------------------------------------------
# ELM PREDICTION FUNCTION
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
# PREDICTION
# ------------------------------------------------

if st.button("Predict Compressive Strength"):

    input_df = pd.DataFrame([[
        material_code,
        replacement,
        binder,
        wb,
        fine_agg,
        coarse_agg,
        sp,
        sio2,
        cao,
        al2o3,
        fe2o3,
        sg,
        wa,
        fm,
        slump,
        t500,
        age
    ]], columns=[
        "Material_Name",
        "%Replace",
        "Binder",
        "w/b",
        "Fine_Agg",
        "Coarse_Agg",
        "SP",
        "SiO2",
        "CaO",
        "Al2O3",
        "Fe2O3",
        "Material_SG",
        "Material_WA",
        "Material_FM",
        "Slump",
        "T50",
        "Age"
    ])

    # Debug (optional)
    st.write("Input Used for Prediction")
    st.write(input_df)

    results = []

    for name, model in models.items():
        pred = elm_predict(input_df, model)[0]
        results.append([name, round(pred, 2)])

    results_df = pd.DataFrame(results, columns=["Model", "Predicted Strength (MPa)"])

    st.subheader("Model Predictions")
    st.dataframe(results_df)

    # ------------------------------------------------
    # MODEL PERFORMANCE (R² BASED)
    # ------------------------------------------------

    model_r2 = {
        "ELM_CMAES": 0.95,
        "ELM_GWO": 0.88,
        "ELM_PSO": 0.91,
        "ELM_WOA": 0.89
    }

    best_model = max(model_r2, key=model_r2.get)

    st.success(f"Best Model (Based on R²): {best_model}")
