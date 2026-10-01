import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Used Car Price Predictor",
    page_icon="🚗",
    layout="centered",
)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = pickle.load(open("car_price_model.pkl", "rb"))
    model_columns = pickle.load(open("model_columns.pkl", "rb"))
    return model, model_columns

model, model_columns = load_model()

CURRENT_YEAR = datetime.now().year

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🚗 Used Car Price Predictor")
st.caption("Estimate the resale value of a used car in India using a Random Forest model (R² > 0.94).")

st.divider()

col1, col2 = st.columns(2)

with col1:
    brand = st.selectbox(
        "Car Brand",
        [
            "Maruti", "Hyundai", "Honda", "Toyota", "Mahindra", "Tata", "Ford",
            "Volkswagen", "Skoda", "Renault", "Nissan", "Kia", "MG",
            "Mercedes-Benz", "BMW", "Audi", "Jaguar", "Land Rover", "Volvo",
        ],
    )
    year = st.number_input("Manufacturing Year", min_value=2000, max_value=CURRENT_YEAR, value=2015, step=1)
    fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])

with col2:
    seats = st.selectbox("Seating Capacity", [5, 7, 8], index=0)
    km_driven = st.number_input("KM Driven", min_value=0, max_value=500_000, value=50_000, step=1_000)
    transmission = st.selectbox("Transmission", ["Manual", "Automatic"])

owner = st.selectbox(
    "Owner Type",
    ["First Owner", "Second Owner", "Third Owner"],
)

# Advanced options (defaults match training dataset medians)
with st.expander("⚙️ Advanced Options (optional — defaults used if unchanged)"):
    adv_col1, adv_col2 = st.columns(2)
    with adv_col1:
        engine = st.number_input("Engine CC", min_value=600, max_value=5000, value=1248, step=50)
        max_power = st.number_input("Max Power (bhp)", min_value=30, max_value=600, value=74, step=5)
    with adv_col2:
        mileage = st.number_input("Mileage (km/l)", min_value=5.0, max_value=40.0, value=20.0, step=0.5)
        torque = st.number_input("Torque (Nm)", min_value=50, max_value=800, value=170, step=10)
    seller_type = st.selectbox("Seller Type", ["Individual", "Dealer", "Trustmark Dealer"])

st.divider()

# ── Predict ───────────────────────────────────────────────────────────────────
if st.button("🔍 Predict Price", use_container_width=True, type="primary"):

    car_age = max(CURRENT_YEAR - year, 1)   # floor at 1 to avoid division by zero
    km_per_year = km_driven / car_age

    data = pd.DataFrame([{
        "Brand":        brand,
        "Seats":        seats,
        "Year":         year,
        "KM_Driven":    km_driven,
        "Fuel":         fuel,
        "Transmission": transmission,
        "Owner":        owner,
        "Seller_Type":  seller_type,
        "Engine":       engine,
        "Max_Power":    max_power,
        "Mileage":      mileage,
        "Torque":       torque,
    }])

    data["Seats"]      = pd.to_numeric(data["Seats"], errors="coerce").fillna(5)
    data["Car_Age"]    = car_age
    data["KM_per_Year"] = km_per_year
    data = data.drop(["Year"], axis=1)

    data = pd.get_dummies(data, drop_first=True)
    data = data.reindex(columns=model_columns, fill_value=0)

    prediction = model.predict(data)[0]
    prediction = max(prediction, 0)   # prices can't be negative

    tree_preds = np.array([tree.predict(data.values)[0] for tree in model.estimators_])
    pred_std   = np.std(tree_preds)
    confidence = max(0.0, min(100.0, 100 - (pred_std / prediction * 100))) if prediction > 0 else 0.0

    low  = max(0, int(prediction - pred_std))
    high = int(prediction + pred_std)

    st.success(f"### 💰 Estimated Price: ₹ {int(prediction):,}")

    r1, r2, r3 = st.columns(3)
    r1.metric("Model Confidence", f"{confidence:.1f}%")
    r2.metric("Lower Estimate",   f"₹ {low:,}")
    r3.metric("Upper Estimate",   f"₹ {high:,}")

    st.progress(int(confidence), text=f"Confidence: {confidence:.1f}%")

    with st.expander("📊 Key pricing factors"):
        st.markdown(
            f"""
| Factor | Your Input | Impact |
|---|---|---|
| Brand | {brand} | Highest — luxury brands 3–4× more |
| Car Age | {car_age} yr(s) | ~10% depreciation per year |
| KM/Year | {km_per_year:,.0f} km | Higher usage → lower value |
| Seats | {seats} | 7/8-seaters carry ₹2–4L premium |
"""
        )
