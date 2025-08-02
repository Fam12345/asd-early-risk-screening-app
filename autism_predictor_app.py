import streamlit as st
import pandas as pd
import joblib
import json

# Load model and supporting artifacts
model = joblib.load("autism_random_forest_model.pkl")
scaler = joblib.load("autism_scaler.pkl")
with open("feature_columns.json") as f:
    feature_columns = json.load(f)

st.set_page_config(page_title="Autism Predictor", layout="wide")
st.title("🧠 Comprehensive Autism Screening Tool")
st.markdown("**Note:** This tool is for educational purposes only and not a substitute for professional diagnosis.")

# Define all inputs
inputs = {}

inputs["Age"] = st.slider("Age", 1, 18, 5)
inputs["Sex"] = st.selectbox("Sex", ["Male", "Female"])
inputs["Home_Language"] = st.selectbox("Primary Home Language", ["English", "Non-English"])
inputs["Race"] = st.selectbox("Race", ["White", "Black", "Asian", "Other"])
inputs["Hispanic_Ethnicity"] = st.selectbox("Hispanic Ethnicity", ["Yes", "No"])
inputs["Born_in_US"] = st.selectbox("Born in US?", ["Yes", "No"])
inputs["Poverty_Level"] = st.selectbox("Poverty Level", ["<100%", "100-199%", "200-399%", "400%+"])
inputs["Family_Size"] = st.slider("Family Size", 1, 10, 4)
inputs["Adverse_Childhood_Exp"] = st.slider("Adverse Childhood Experiences (count)", 0, 10, 1)
inputs["Parental_Education"] = st.selectbox("Parental Education", ["Less than high school", "High school", "Some college", "College or more"])
inputs["Housing_Instability"] = st.selectbox("Housing Instability", ["Yes", "No"])
inputs["Food_Insecurity"] = st.selectbox("Food Insecurity", ["Yes", "No"])
inputs["ADHD_Diagnosis"] = st.selectbox("Diagnosed with ADHD?", ["Yes", "No"])
inputs["Anxiety_Diagnosis"] = st.selectbox("Diagnosed with Anxiety?", ["Yes", "No"])
inputs["Dev_Delay_Diagnosis"] = st.selectbox("Developmental Delay Diagnosis?", ["Yes", "No"])
inputs["Breastfed"] = st.selectbox("Was the child breastfed?", ["Yes", "No"])
inputs["Allergies"] = st.selectbox("Has allergies?", ["Yes", "No"])
inputs["Seizures"] = st.selectbox("Has seizures?", ["Yes", "No"])
inputs["Heart_Condition"] = st.selectbox("Heart condition?", ["Yes", "No"])
inputs["Asthma"] = st.selectbox("Asthma?", ["Yes", "No"])
inputs["Premature_Birth"] = st.selectbox("Premature birth?", ["Yes", "No"])
inputs["Low_Birthweight"] = st.selectbox("Low birthweight?", ["Yes", "No"])
inputs["BMI_Category"] = st.selectbox("BMI Category", ["Underweight", "Normal", "Overweight", "Obese"])
inputs["Speech_Concern"] = st.selectbox("Concerned about speech?", ["Yes", "No"])
inputs["Interaction_Concern"] = st.selectbox("Concerned about interaction?", ["Yes", "No"])
inputs["WordPhrase_Concern"] = st.selectbox("Concerned about word/phrase use?", ["Yes", "No"])
inputs["Behavior_Concern"] = st.selectbox("Concerned about behavior?", ["Yes", "No"])
inputs["Says_One_Word"] = st.selectbox("Says at least one word?", ["Yes", "No"])
inputs["Says_Two_Words"] = st.selectbox("Says at least two words?", ["Yes", "No"])
inputs["Asks_Questions"] = st.selectbox("Asks questions?", ["Yes", "No"])
inputs["Screen_Time_Hours"] = st.slider("Daily screen time (hours)", 0, 12, 2)
inputs["Sleep_Hours"] = st.slider("Average sleep (hours)", 0, 14, 9)
inputs["Maternal_Mental_Health"] = st.selectbox("Maternal mental health", ["Good", "Average", "Poor"])
inputs["Neighborhood_Support"] = st.selectbox("Neighborhood support", ["Strong", "Moderate", "Weak"])
inputs["Parental_Aggravation"] = st.selectbox("Parental aggravation?", ["Often", "Sometimes", "Rarely"])
inputs["Family_Resilience"] = st.selectbox("Family resilience?", ["High", "Medium", "Low"])
inputs["Income_Group"] = st.selectbox("Income group", ["Low", "Medium", "High"])

# Mapping to model input
def encode_binary(val):
    return 1 if val == "Yes" else 0

encoded = {
    "Sex": 1 if inputs["Sex"] == "Male" else 0,
    "Home_Language": 0 if inputs["Home_Language"] == "English" else 1,
    "Speech_Concern": encode_binary(inputs["Speech_Concern"]),
    "Interaction_Concern": encode_binary(inputs["Interaction_Concern"]),
    "WordPhrase_Concern": encode_binary(inputs["WordPhrase_Concern"]),
    "Behavior_Concern": encode_binary(inputs["Behavior_Concern"]),
    "Says_One_Word": encode_binary(inputs["Says_One_Word"]),
    "Says_Two_Words": encode_binary(inputs["Says_Two_Words"]),
    "Asks_Questions": encode_binary(inputs["Asks_Questions"]),
    "Breastfed": encode_binary(inputs["Breastfed"]),
    "Allergies": encode_binary(inputs["Allergies"]),
    "Seizures": encode_binary(inputs["Seizures"]),
    "Heart_Condition": encode_binary(inputs["Heart_Condition"]),
    "Asthma": encode_binary(inputs["Asthma"]),
    "Premature_Birth": encode_binary(inputs["Premature_Birth"]),
    "Low_Birthweight": encode_binary(inputs["Low_Birthweight"]),
    "Housing_Instability": encode_binary(inputs["Housing_Instability"]),
    "Food_Insecurity": encode_binary(inputs["Food_Insecurity"]),
    "Born_in_US": encode_binary(inputs["Born_in_US"]),
    "ADHD_Diagnosis": encode_binary(inputs["ADHD_Diagnosis"]),
    "Anxiety_Diagnosis": encode_binary(inputs["Anxiety_Diagnosis"]),
    "Dev_Delay_Diagnosis": encode_binary(inputs["Dev_Delay_Diagnosis"]),
}

# Add remaining numeric and label fields
for col in feature_columns:
    if col not in encoded:
        encoded[col] = inputs.get(col, 0)

# Prepare final dataframe
input_df = pd.DataFrame([encoded])[feature_columns]

if st.button("🔍 Predict Autism Likelihood"):
    try:
        scaled = scaler.transform(input_df)
        prediction = model.predict(scaled)[0]
        proba = model.predict_proba(scaled)[0][1]

        if prediction == 1:
            st.error(f"🧩 High likelihood of autism (Confidence: {proba:.2%})")
        else:
            st.success(f"✅ Low likelihood of autism (Confidence: {1 - proba:.2%})")

    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
