import streamlit as st
import numpy as np
import joblib

# Load the trained LightGBM model
model = joblib.load("LGBM.pkl")  # Load LightGBM model

# Streamlit UI
st.title("Bank Nifty Price Prediction (LightGBM)")

st.write("Enter the last 5 prices of Bank Nifty:")

# Create input fields for 5 prices
prices = []
for i in range(5):
    price = st.number_input(f"Price {i+1}", min_value=0.0, format="%.2f")
    prices.append(price)

# Predict button
if st.button("Predict"):
    if all(p > 0 for p in prices):  # Ensure all inputs are valid
        # Convert input to NumPy array & reshape correctly for LightGBM
        prices_array = np.array(prices, dtype=float).reshape(1, -1)

        # Predict using LightGBM model
        prediction = model.predict(prices_array)[0]  # Get single value

        st.success(f"Predicted Price: {prediction:.2f}")
    else:
        st.error("Please enter valid prices for all fields.")
