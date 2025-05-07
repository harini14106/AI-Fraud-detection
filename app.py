import streamlit as st
import numpy as np
import pickle

# Load the trained model
model = pickle.load(open("D:/ai fraud detection/model.pkl", "rb"))

# UI layout
st.set_page_config(page_title="Fraud Detection", page_icon="🛡️", layout="centered")
st.title("🛡️ AI-Powered Fraud Detection System")
st.markdown("Enter transaction details below:")

# Input fields
amount = st.number_input("Transaction Amount", min_value=0.0)
time = st.number_input("Transaction Time", min_value=0.0)

# Add 28 anonymized V-features (for demo, default is 0.0)
v_inputs = []
for i in range(1, 29):
    val = st.number_input(f"V{i}", value=0.0, key=f"V{i}")
    v_inputs.append(val)

# Predict button
if st.button("Predict"):
    features = np.array([time, amount] + v_inputs).reshape(1, -1)
    prediction = model.predict(features)[0]
    st.subheader("🔎 Prediction:")
    if prediction == 0:
        st.success("✅ Transaction is NOT Fraudulent")
    else:
        st.error("⚠️ Transaction is FRAUDULENT")
