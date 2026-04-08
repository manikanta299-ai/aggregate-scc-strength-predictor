import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="SCC Strength Predictor", layout="wide")

st.title("AI-Based SCC Compressive Strength Prediction")
st.markdown("### Using Optimized ELM-CMAES Model")

# ------------------------------------------------
# MATERIALS (FCS REMOVED)
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
# LOAD BEST MODEL ONLY
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
# INPUT SECTION
# ------------------------------------------------

st.subheader("Input Parameters")

col1, col2, col3 = st.columns(3)

with col1:
    material = st.selectbox("Material Type", material_names)
    replacement = st.slider("% Replacement", 0.0, 100.0, 10.0)

with col2:
    binder = st.number_input("Binder (kg/m³)", value=400.0)
    wb = st.number_input("Water/Binder Ratio", value=0.55)

with col3:
    fine_agg = st.number_input("Fine Aggregate", value=829.6)
    coarse_agg = st.number_input("Coarse Aggregate", value=656.0)

col4, col5, col6 = st.columns(3)

with col4:
    sp = st.number_input("Superplasticizer (%)", value=1.05)
    sio2 = st.number_input("SiO2 (%)", value=30.2)

with col5:
    cao = st.number_input("CaO (%)", value=39.5)
    al2o3 = st.number_input("Al2O3 (%)", value=3.0)

with col6:
    fe2o3 = st.number_input("Fe2O3 (%)", value=31.8)
    sg = st.number_input("Material Specific Gravity", value=2.45)

col7, col8, col9 = st.columns(3)

with col7:
    wa = st.number_input("Water Absorption (%)", value=8.0)

with col8:
    fm = st.number_input("Fineness Modulus", value=3.11)

with col9:
    slump = st.number_input("Slump (mm)", value=705.0)

col10, col11 = st.columns(2)

with col10:
    t500 = st.number_input("T500 (sec)", value=3.4)

with col11:
    age = st.number_input("Age (days)", value=7)

material_code = material_reverse[material]

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

    prediction = elm_predict(input_df, model)[0]

    # ------------------------------------------------
    # OUTPUT SECTION
    # ------------------------------------------------

    st.subheader("Prediction Result")

    st.success(f"Predicted Compressive Strength: {prediction:.2f} MPa")

    # Interpretation
    if prediction < 30:
        st.warning("Low Strength Mix")
    elif prediction < 60:
        st.info("Moderate Strength Mix")
    else:
        st.success("High Strength SCC")

    # ------------------------------------------------
    # INPUT SUMMARY
    # ------------------------------------------------

    with st.expander("View Input Summary"):
        st.dataframe(input_df)

    # ------------------------------------------------
    # OPTIONAL VALIDATION
    # ------------------------------------------------

    actual = st.number_input("Enter Actual Strength (Optional)", value=0.0)

    if actual > 0:
        error = abs(prediction - actual)

        st.subheader("Prediction Error")
        st.write(f"Absolute Error: {error:.2f} MPa")

        if error < 2:
            st.success("Excellent Prediction Accuracy")
        elif error < 5:
            st.info("Good Prediction")
        else:
            st.warning("Prediction deviation is high")
