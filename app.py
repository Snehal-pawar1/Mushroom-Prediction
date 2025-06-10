# app.py
import streamlit as st
import pandas as pd
import joblib
import os

# Load saved model and encoders
model = joblib.load("random_forest_model.pkl")
label_y = joblib.load("label_y_encoder.pkl")

# Load dataset to get column names and possible values
df = pd.read_csv("converted_mushroom_dataset.csv")
X = df.drop("class", axis=1)

# Load encoders
encoders = {}
for col in X.columns:
    try:
        encoders[col] = joblib.load(f"encoder_{col}.pkl")
    except FileNotFoundError:
        pass

# UI
st.title("🍄 Mushroom Edibility Predictor")
st.write("Predict whether a mushroom is **edible** or **poisonous** based on its features.")

st.subheader("Enter Mushroom Characteristics:")
input_data = {}
for col in X.columns:
    if col in encoders:
        options = encoders[col].classes_
        choice = st.selectbox(f"{col}", options)
        encoded = encoders[col].transform([choice])[0]
        input_data[col] = encoded
    else:
        value = st.number_input(f"{col}", min_value=0, max_value=100, value=10)
        input_data[col] = value

# Prediction
if st.button("Predict"):
    input_df = pd.DataFrame([input_data])
    pred = model.predict(input_df)[0]
    result = label_y.inverse_transform([pred])[0]
    
    if result == "poisonous":
        st.error("❌ This mushroom is likely **Poisonous**.")
    else:
        st.success("✅ This mushroom is likely **Edible**.")
