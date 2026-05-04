import streamlit as st
import pickle
import numpy as np

st.title("📊 Telecom Churn Prediction App")

# load model + scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# simple inputs
account_length = st.number_input("Account Length", 1, 300, 100)
intl_plan = st.selectbox("International Plan (0=No, 1=Yes)", [0, 1])
vmail = st.number_input("Voice Mail Messages", 0, 50, 0)
day_minutes = st.number_input("Day Minutes", 0.0, 400.0, 200.0)
service_calls = st.number_input("Customer Service Calls", 0, 10, 1)

# button
if st.button("Predict"):
    data = np.array([[account_length, intl_plan, vmail, day_minutes, service_calls]])

    data_scaled = scaler.transform(data)
    pred = model.predict(data_scaled)

    if pred[0] == 1:
        st.error("❌ Customer will CHURN")
    else:
        st.success("✅ Customer will STAY")