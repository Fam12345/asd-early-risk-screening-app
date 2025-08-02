# autism_predictor_app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load your trained model and scaler
model = joblib.load("autism_random_forest_model.pkl")
scaler = joblib.load("autism_scaler.pkl")

st.set_page_config(page_title="Autism Screening", layout="centered")

# App title and disclaimer
st.title("🧠 Early Autism Screening Tool")

st.markdown(
    """
    **⚠️ Disclaimer:**  
    This tool is for **educational and awareness purposes only**.  
    It does **not provide a medical diagnosis**.  
    Please consult a licensed pediatrician, psychologist, or developmental specialist for any concerns about autism.
    """,
    unsafe_allow_html=True
)

# Define user input fields
st.header("📋 Child Information")

age = st.slider("Age of child (in years)", min_value=0, max_value=17, value=5)
sex = st.selectbox("Sex", ["Male", "Female"])
home_lang = st.selectbox("Primary home language", ["English", "Non-English"])
parent_edu = st.selectbox(
    "Highest parental education",
    ["Less than high school", "High school", "Some college", "College or more"]
)
sleep_hours = st.slider("Average sleep hours per night", 0, 14, 9)
screen_hours = st.slider("Average screen time per day (hours)", 0, 10, 2)

st.header("⚕️ Health & Behavior")

speech_concern = st.selectbox("Concern: Speech delay?", ["No", "Yes"])
interaction_concern = st.selectbox("Concern: Social interaction?", ["No", "Yes"])
word_phrase_concern = st.selectbox("Concern: Use of words/phrases?", ["No", "Yes"])
maternal_mental_health = st.selectbox(
    "Maternal mental health",
    ["Good", "Average", "Poor"]
)

# Prepare input data
if st.button("🔍 Predict Autism Likelihood"):

    # Map inputs
    input_dict = {
        "Age": age,
        "Sex_Label": 1 if sex == "Male" else 0,
        "Home_Language_Label": 0 if home_lang == "English" else 1,
        "Parental_Education_Label": {
            "Less than high school": 0,
            "High school": 1,
            "Some college": 2,
            "College or more": 3
        }[parent_edu],
        "Sleep_Hours": sleep_hours,
        "Screen_Time_Hours": screen_hours,
        "Speech_Concern": 1 if speech_concern == "Yes" else 0,
        "Interaction_Concern": 1 if interaction_concern == "Yes" else 0,
        "WordPhrase_Concern": 1 if word_phrase_concern == "Yes" else 0,
        "Maternal_Mental_Health_Label": {
            "Good": 0,
            "Average": 1,
            "Poor": 2
        }[maternal_mental_health],
    }

    input_df = pd.DataFrame([input_dict])

    # Scale
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0][1]

    # Display result
    if prediction == 1:
        st.error(f"🧩 High likelihood of autism. (Confidence: {proba:.2%})")
    else:
        st.success(f"✅ Low likelihood of autism. (Confidence: {1 - proba:.2%})")

    st.markdown("---")
    st.markdown("Remember: This tool does not replace professional medical advice.")
