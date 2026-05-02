import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model
model = pickle.load(open("car_price_model.pkl", "rb"))
model_columns = pickle.load(open("model_columns.pkl", "rb"))

st.title("🚗 Used Car Price Predictor")


# User inputs - KEY FEATURES
# 1. Brand = Highest impact feature (3-4x price difference)
brand = st.selectbox(
    "Car Brand *",
    ["Maruti", "Hyundai", "Honda", "Toyota", "Mahindra", "Tata", "Ford",
     "Volkswagen", "Skoda", "Renault", "Nissan", "Kia", "MG",
     "Mercedes-Benz", "BMW", "Audi", "Jaguar", "Land", "Volvo"]
)

# 2. Seats = Important for larger vehicles (+₹200K-400K premium)
seats = st.selectbox("Seating Capacity *", [5, 7, 8], index=0)

# 3. Car Age & KM/Year = Strong depreciation factors
year = st.number_input("Year *", 2000, 2024, 2015)
km_driven = st.number_input("KM Driven *", 0, 300000, 50000)

# Other features
fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
owner = st.selectbox("Owner Type", ["First Owner", "Second Owner", "Third Owner"])

# Predict button
if st.button("Predict Price"):

    data = pd.DataFrame([{
        "Brand": brand,  # Brand directly from dropdown
        "Seats": seats,
        "Year": year,
        "KM_Driven": km_driven,
        "Fuel": fuel,
        "Transmission": transmission,
        "Owner": owner,
        "Seller_Type": "Individual",  # Default value
        "Engine": 1248,      # Default values
        "Max_Power": 74,
        "Mileage": 20.0,
        "Torque": 170
    }])

    # Feature engineering - MUST MATCH TRAINING
    data['Seats'] = pd.to_numeric(data['Seats'], errors='coerce').fillna(5)
    data['Car_Age'] = 2024 - data['Year']
    data['KM_per_Year'] = data['KM_Driven'] / data['Car_Age']

    data = data.drop(['Year'], axis=1)

    # Encode categorical variables
    data = pd.get_dummies(data, drop_first=True)

    # Match training columns - CRITICAL STEP
    data = data.reindex(columns=model_columns, fill_value=0)

    # Prediction with confidence
    prediction = model.predict(data)[0]

    # Get predictions from all trees for confidence interval
    tree_preds = np.array([tree.predict(data.values)[0] for tree in model.estimators_])
    pred_std = np.std(tree_preds)
    confidence = max(0, 100 - (pred_std / prediction * 100))

    # Display prediction with confidence
    st.success(f"💰 Estimated Car Price: ₹ {int(prediction):,}")

    # Show confidence metric
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Model Confidence", f"{confidence:.1f}%")
    with col2:
        st.metric("Price Range", f"₹{int(prediction - pred_std):,} - ₹{int(prediction + pred_std):,}")

    # Show feature importance context
    st.info("**Key pricing factors used:**\n"
            "1. **Brand**: Luxury brands cost 3-4x more\n"
            "2. **Seats**: 7/8-seaters have ₹200K-400K premium\n"
            "3. **Car Age**: Each year = ~10% depreciation\n"
            "4. **KM/Year**: Higher usage = lower price")