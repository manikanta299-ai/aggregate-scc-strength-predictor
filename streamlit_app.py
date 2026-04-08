import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="SCC Strength Predictor (ELM)", layout="wide")

st.title("AI-Based SCC Compressive Strength Prediction (ELM Models)")

# ------------------------------------------------
# MATERIAL MAPPING (CODE → NAME)
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

# Reverse mapping
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
    replacement = st.number_input("% Replacement", value=20.0)
    binder = st.number_input("Binder (kg/m³)", value=400.0)

with col2:
    wb = st.number_input("Water/Binder Ratio", value=0.40)
    fine_agg = st.number_input("Fine Aggregate", value=700.0)
    coarse_agg = st.number_input("Coarse Aggregate", value=1000.0)

with col3:
    sp = st.number_input("Superplasticizer (%)", value=1.5)
    sio2 = st.number_input("SiO2 (%)", value=60.0)
    cao = st.number_input("CaO (%)", value=10.0)

col4, col5, col6 = st.columns(3)

with col4:
    al2o3 = st.number_input("Al2O3 (%)", value=5.0)
    fe2o3 = st.number_input("Fe2O3 (%)", value=3.0)

with col5:
    sg = st.number_input("Material Specific Gravity", value=2.6)
    wa = st.number_input("Water Absorption (%)", value=2.0)

with col6:
    fm = st.number_input("Fineness Modulus", value=2.8)
    slump = st.number_input("Slump (mm)", value=650.0)

col7, col8 = st.columns(2)

with col7:
    t500 = st.number_input("T500 (sec)", value=4.0)

with col8:
    age = st.number_input("Age (days)", value=28)

# Encode material
material_code = material_reverse[material]
# ------------------------------------------------
# ELM PREDICTION FUNCTION
# ------------------------------------------------

def elm_predict(X, model_dict):

    W = model_dict["W"]
    b = model_dict["b"]
    beta = model_dict["beta"]
    scaler = model_dict["scaler"]

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
        "%Replacement",
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
    
    st.write("INPUT VALUES FROM GUI")
    st.write(input_df)
    
    results = []

    for name, model in models.items():

        pred = elm_predict(input_df, model)[0]

        results.append([name, round(pred, 2)])

    results_df = pd.DataFrame(
        results,
        columns=["Model", "Predicted Strength (MPa)"]
    )

    st.subheader("Model Predictions")

    st.dataframe(results_df)

    best_row = results_df.loc[results_df["Predicted Strength (MPa)"].idxmax()]

    st.success(f"Best Model: {best_row['Model']} | Strength = {best_row['Predicted Strength (MPa)']} MPa")
