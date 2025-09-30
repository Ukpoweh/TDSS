import streamlit as st
import pandas as pd
import pickle

# -------------------------------
# Load trained pipeline & label encoder
# -------------------------------
MODEL_PATH = "models_classification/final_model_pipeline.pkl"
LE_PATH = "models_classification/label_encoder.pkl"

model = pickle.load(MODEL_PATH)
try:
    le = pickle.load(LE_PATH)  # if saved separately
except:
    le = None

st.set_page_config(page_title="Pavement Rehabilitation TDSS", layout="wide")

# -------------------------------
# App Title
# -------------------------------
st.title("🛣️ Pavement Rehabilitation TDSS")
st.write("""
This ML-powered **Transportation Decision Support System (TDSS)** assists civil engineers in 
selecting **optimal pavement rehabilitation materials** based on soil, traffic, environmental, 
geospatial, and structural inputs.
""")

# -------------------------------
# Mode Selection
# -------------------------------
mode = st.radio("Choose Prediction Mode:", ["Single Project Input", "Batch Upload (CSV)"])

# =====================================================
# SINGLE INPUT MODE
# =====================================================
if mode == "Single Project Input":
    st.sidebar.header("Enter Project Inputs")

    # Numeric inputs
    latitude = st.sidebar.number_input("Latitude", value=7.50, format="%.4f")
    longitude = st.sidebar.number_input("Longitude", value=4.50, format="%.4f")
    lane = st.sidebar.number_input("Number of Lanes", min_value=1, step=1, value=2)
    bulk_density = st.sidebar.number_input("Bulk Density (g/cm³)", min_value=1.0, max_value=3.0, step=0.01, value=2.10)
    clay = st.sidebar.slider("Clay %", 0, 100, 30)
    sand = st.sidebar.slider("Sand %", 0, 100, 40)
    silt = st.sidebar.slider("Silt %", 0, 100, 30)
    ph = st.sidebar.number_input("Soil pH", min_value=0.0, max_value=14.0, step=0.1, value=7.0)
    temperature = st.sidebar.number_input("Average Temperature (°C)", min_value=-10.0, max_value=50.0, step=0.5, value=28.0)
    precipitation = st.sidebar.number_input("Annual Precipitation (mm)", min_value=0.0, step=10.0, value=1200.0)
    length = st.sidebar.number_input("Approximate Length (km)", min_value=0.0, step=0.1, value=5.0)
    cbr = st.sidebar.number_input("Estimated CBR (%)", min_value=0.0, max_value=100.0, step=1.0, value=30.0)
    adt = st.sidebar.number_input("Average Daily Traffic (veh/day)", min_value=0, step=100, value=1000)
    esal = st.sidebar.number_input("20-year ESAL", min_value=0.0, step=100.0, value=50000.0)
    pavement_thickness = st.sidebar.number_input("Pavement Thickness (mm)", min_value=0.0, step=1.0, value=150.0)
    esal_log = st.sidebar.number_input("ESAL 20yr (log)", min_value=0.0, step=0.01, value=10.8)

    # Categorical inputs
    state = st.sidebar.text_input("State", value="Ondo")
    road_type = st.sidebar.selectbox("Road Type", ["trunk", "primary", "secondary"])
    surface = st.sidebar.selectbox("Paved?", ["Paved", "Not Paved"])
    dominant_texture = st.sidebar.selectbox("Dominant Soil Texture", ["clay", "sand", "silt"])

    # Collect inputs
    input_data = pd.DataFrame([{
        "latitude": latitude,
        "longitude": longitude,
        "lane": lane,
        "bulk_density_g_cm³": bulk_density,
        "clay_pct": clay,
        "sand_pct": sand,
        "silt_pct": silt,
        "ph": ph,
        "avg_temperature_°c": temperature,
        "annual_precipitation_mm": precipitation,
        "approximate_length_km": length,
        "estimated_cbr_pct": cbr,
        "adt_veh_day": adt,
        "esal_20yr": esal,
        "pavement_thickness_mm": pavement_thickness,
        "esal_20yr_log": esal_log,
        "state": state,
        "type": road_type,
        "surface": surface,
        "dominant_texture": dominant_texture
    }])

    st.subheader("📊 Project Input Summary")
    st.dataframe(input_data)

    if st.button("Predict Rehabilitation Material"):
        prediction = model.predict(input_data)[0]

        if le:
            prediction_label = le.inverse_transform([prediction])[0]
        else:
            prediction_label = prediction

        st.success(f"✅ Recommended Rehabilitation Material: **{prediction_label}**")

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(input_data)[0]
            prob_df = pd.DataFrame({
                "Material": le.classes_ if le else range(len(probs)),
                "Probability": probs
            }).sort_values("Probability", ascending=False)
            st.subheader("🔎 Prediction Probabilities")
            st.bar_chart(prob_df.set_index("Material"))

# =====================================================
# BATCH UPLOAD MODE
# =====================================================
elif mode == "Batch Upload (CSV)":
    st.subheader("📂 Upload a CSV file with project data")

    st.markdown("""
    **Required columns (must match training features):**  
    - Numeric: `latitude`, `longitude`, `lane`, `bulk_density_g_cm³`, `clay_pct`, `sand_pct`, `silt_pct`,  
      `ph`, `avg_temperature_°c`, `annual_precipitation_mm`, `approximate_length_km`,  
      `estimated_cbr_pct`, `adt_veh_day`, `esal_20yr`, `pavement_thickness_mm`, `esal_20yr_log`  
    - Categorical: `state`, `type`, `surface`, `dominant_texture`
    """)

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        df_batch = pd.read_csv(uploaded_file)
        st.write("✅ Uploaded Data Preview:")
        st.dataframe(df_batch.head())

        if st.button("Run Batch Predictions"):
            preds = model.predict(df_batch)

            if le:
                preds_labels = le.inverse_transform(preds)
            else:
                preds_labels = preds

            df_batch["Predicted_Material"] = preds_labels
            st.success("✅ Predictions Complete")
            st.dataframe(df_batch)

            # Option to download results
            csv_download = df_batch.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Predictions as CSV",
                data=csv_download,
                file_name="tdss_predictions.csv",
                mime="text/csv"
            )

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("Developed as part of Pavement Rehabilitation ML-based TDSS project (Nigeria case study).")
