import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="SCC Strength Predictor", layout="wide")

# -------------------------------
# TITLE
# -------------------------------
st.title("AI-Based aggregate SCC Compressive Strength Prediction")
st.markdown("### Optimized ELM-CMAES Model (Best Performing Model)")

# -------------------------------
# MATERIAL MAP (FCS REMOVED)
# -------------------------------
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

# -------------------------------
# LOAD MODEL
# -------------------------------
model = joblib.load("ELM_CMAES.pkl")

# -------------------------------
# ELM FUNCTION
# -------------------------------
def elm_predict(X_df, model_dict):
    W = model_dict["W"]
    b = model_dict["b"]
    beta = model_dict["beta"]
    scaler = model_dict["scaler"]

    X = np.array(X_df).reshape(1, -1)
    X_scaled = scaler.transform(X)

    H = 1 / (1 + np.exp(-(X_scaled @ W + b)))

    return H @ beta

# -------------------------------
# INPUT UI
# -------------------------------
st.subheader("🔧 Input Parameters")

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

# -------------------------------
# PREDICTION
# -------------------------------
if st.button("🚀 Predict Compressive Strength"):

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
        "Material_Name", "%Replace", "Binder", "w/b",
        "Fine_Agg", "Coarse_Agg", "SP",
        "SiO2", "CaO", "Al2O3", "Fe2O3",
        "Material_SG", "Material_WA", "Material_FM",
        "Slump", "T50", "Age"
    ])

    prediction = elm_predict(input_df, model)[0]

    # -------------------------------
    # RESULT DISPLAY
    # -------------------------------
    st.subheader("📊 Prediction Result")

    st.success(f"Predicted Strength: {prediction:.2f} MPa")

    if prediction < 30:
        st.error("LOW STRENGTH MIX")
    elif prediction < 60:
        st.warning("MODERATE STRENGTH MIX")
    else:
        st.success("HIGH STRENGTH SCC")

    # -------------------------------
    # INPUT SUMMARY
    # -------------------------------
    with st.expander("📄 View Input Summary"):
        st.dataframe(input_df)

    # -------------------------------
    # SENSITIVITY ANALYSIS
    # -------------------------------
    st.subheader("📈 Replacement Sensitivity")

    rep_range = np.linspace(0, 60, 30)
    preds = []

    for r in rep_range:
        temp = input_df.copy()
        temp["%Replace"] = r
        preds.append(elm_predict(temp, model)[0])

    fig1 = plt.figure()
    plt.plot(rep_range, preds, linewidth=2)
    plt.xlabel("Replacement (%)")
    plt.ylabel("Strength (MPa)")
    plt.title("Effect of Replacement")
    plt.grid()
    st.pyplot(fig1)

    # -------------------------------
    # FEATURE IMPACT
    # -------------------------------
    st.subheader("📊 Feature Influence")

    features = input_df.columns
    base_pred = prediction
    impacts = []

    for col in features:
        temp = input_df.copy()
        temp[col] = temp[col] * 1.05
        impacts.append(elm_predict(temp, model)[0] - base_pred)

    fig2 = plt.figure()
    plt.barh(features, impacts)
    plt.xlabel("Impact on Strength")
    plt.title("Feature Sensitivity (+5%)")
    plt.grid()
    st.pyplot(fig2)

    # -------------------------------
    # OPTIONAL VALIDATION
    # -------------------------------
    st.subheader("🎯 Validation (Optional)")

    actual = st.number_input("Enter Actual Strength", value=0.0)

    if actual > 0:
        error = abs(prediction - actual)

        st.write(f"Error: {error:.2f} MPa")

        if error < 2:
            st.success("Excellent Prediction")
        elif error < 5:
            st.info("Good Prediction")
        else:
            st.warning("High Deviation")
