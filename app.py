import streamlit as st
import pandas as pd
import joblib

# -------------------------------
# Load the trained pipeline & label encoder
# -------------------------------
MODEL_PATH = "models_classification/final_model_pipeline.pkl"
LE_PATH = "models_classification/label_encoder.pkl"

model = joblib.load(MODEL_PATH)
try:
    le = joblib.load(LE_PATH)  # if you saved LabelEncoder separately
except:
    le = None

st.set_page_config(page_title="Pavement Rehabilitation TDSS", layout="wide")

# -------------------------------
# App Title
# -------------------------------
st.title("🛣️ Pavement Rehabilitation TDSS")
st.write("""
This system predicts the **optimal pavement rehabilitation material** using 
traffic, soil, climate, geospatial, and structural inputs.
""")

# -------------------------------
# Sidebar Inputs (exact features from pipeline)
# -------------------------------
st.sidebar.header("Project Inputs")

# Numeric features
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

# Categorical features
state = st.sidebar.text_input("State", value="Ondo")
road_type = st.sidebar.selectbox("Road Type", ["Trunk", "Urban", "Rural", "Highway"])
surface = st.sidebar.selectbox("Surface", ["Asphalt", "Gravel", "Concrete"])
dominant_texture = st.sidebar.selectbox("Dominant Soil Texture", ["clay", "sand", "silt"])

# -------------------------------
# Collect inputs into dataframe
# -------------------------------
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

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Rehabilitation Material"):
    prediction = model.predict(input_data)[0]

    # Decode with LabelEncoder if available
    if le:
        prediction_label = le.inverse_transform([prediction])[0]
    else:
        prediction_label = prediction

    st.success(f"✅ Recommended Rehabilitation Material: **{prediction_label}**")

    # Probabilities (if model supports predict_proba)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(input_data)[0]
        prob_df = pd.DataFrame({
            "Material": le.classes_ if le else range(len(probs)),
            "Probability": probs
        }).sort_values("Probability", ascending=False)

        st.subheader("🔎 Prediction Probabilities")
        st.bar_chart(prob_df.set_index("Material"))

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("Developed as part of Pavement Rehabilitation ML-based TDSS project (Nigeria case study).")
