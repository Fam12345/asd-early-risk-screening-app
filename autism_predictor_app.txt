import streamlit as st
import pandas as pd
import joblib
import json

# Load model and scaler
model = joblib.load("autism_random_forest_model.pkl")
scaler = joblib.load("autism_scaler.pkl")

# Load feature columns
with open("feature_columns.json") as f:
    feature_columns = json.load(f)

# Hardcoded encoding map based on your training data
race_map = {"White": 0, "Black": 1, "Asian": 2, "Other": 3}
ethnicity_map = {"Yes": 1, "No": 0}
language_map = {"English": 0, "Non-English": 1}
sex_map = {"Male": 1, "Female": 0}
bmi_map = {"Underweight": 0, "Normal": 1, "Overweight": 2, "Obese": 3}
education_map = {
    "Less than high school": 0,
    "High school": 1,
    "Some college": 2,
    "College or more": 3
}
mental_health_map = {"Good": 0, "Average": 1, "Poor": 2}
support_map = {"Strong": 0, "Moderate": 1, "Weak": 2}
aggravation_map = {"Rarely": 0, "Sometimes": 1, "Often": 2}
resilience_map = {"Low": 0, "Medium": 1, "High": 2}
income_map = {"Low": 0, "Medium": 1, "High": 2}

def encode_binary(val):
    return 1 if val == "Yes" else 0

# Page settings
st.set_page_config(page_title="Autism Predictor", layout="wide")
st.title("🧠 Comprehensive Autism Screening Tool")

st.markdown("**Note:** This tool is for educational purposes only and not a substitute for professional diagnosis.")

st.markdown(
    """
    🩺 **Note:**  
    This tool is built using features derived from the **2023 National Survey of Children’s Health (NSCH)**  
    and remains in a **preliminary research phase**.

    If you are interested in this work or would like to support the research,  
    please search for **Rabia Endris** on LinkedIn.
    """,
    unsafe_allow_html=True
)



# Collect user input
inputs = {}
inputs["Age"] = st.slider("Age", 1, 18, 5)
inputs["Sex"] = st.selectbox("Sex", list(sex_map.keys()))
inputs["Home_Language"] = st.selectbox("Primary Home Language", list(language_map.keys()))
inputs["Race"] = st.selectbox("Race", list(race_map.keys()))
inputs["Hispanic_Ethnicity"] = st.selectbox("Hispanic Ethnicity", list(ethnicity_map.keys()))
inputs["Born_in_US"] = st.selectbox("Born in US?", list(ethnicity_map.keys()))
inputs["Poverty_Level"] = st.selectbox("Poverty Level", ["<100%", "100-199%", "200-399%", "400%+"])
inputs["Family_Size"] = st.slider("Family Size", 1, 10, 4)
inputs["Adverse_Childhood_Exp"] = st.slider("Adverse Childhood Experiences (count)", 0, 10, 1)
inputs["Parental_Education"] = st.selectbox("Parental Education", list(education_map.keys()))
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
inputs["BMI_Category"] = st.selectbox("BMI Category", list(bmi_map.keys()))
inputs["Speech_Concern"] = st.selectbox("Concerned about speech?", ["Yes", "No"])
inputs["Interaction_Concern"] = st.selectbox("Concerned about interaction?", ["Yes", "No"])
inputs["WordPhrase_Concern"] = st.selectbox("Concerned about word/phrase use?", ["Yes", "No"])
inputs["Behavior_Concern"] = st.selectbox("Concerned about behavior?", ["Yes", "No"])
inputs["Says_One_Word"] = st.selectbox("Says at least one word?", ["Yes", "No"])
inputs["Says_Two_Words"] = st.selectbox("Says at least two words?", ["Yes", "No"])
inputs["Asks_Questions"] = st.selectbox("Asks questions?", ["Yes", "No"])
inputs["Screen_Time_Hours"] = st.slider("Daily screen time (hours)", 0, 12, 2)
inputs["Sleep_Hours"] = st.slider("Average sleep (hours)", 0, 14, 9)
inputs["Maternal_Mental_Health"] = st.selectbox("Maternal mental health", list(mental_health_map.keys()))
inputs["Neighborhood_Support"] = st.selectbox("Neighborhood support", list(support_map.keys()))
inputs["Parental_Aggravation"] = st.selectbox("Parental aggravation?", list(aggravation_map.keys()))
inputs["Family_Resilience"] = st.selectbox("Family resilience?", list(resilience_map.keys()))
inputs["Income_Group"] = st.selectbox("Income group", list(income_map.keys()))

# Encode user inputs
encoded = {
    "Sex": sex_map[inputs["Sex"]],
    "Home_Language": language_map[inputs["Home_Language"]],
    "Race": race_map[inputs["Race"]],
    "Hispanic_Ethnicity": ethnicity_map[inputs["Hispanic_Ethnicity"]],
    "Born_in_US": ethnicity_map[inputs["Born_in_US"]],
    "BMI_Category": bmi_map[inputs["BMI_Category"]],
    "Parental_Education": education_map[inputs["Parental_Education"]],
    "Maternal_Mental_Health": mental_health_map[inputs["Maternal_Mental_Health"]],
    "Neighborhood_Support": support_map[inputs["Neighborhood_Support"]],
    "Parental_Aggravation": aggravation_map[inputs["Parental_Aggravation"]],
    "Family_Resilience": resilience_map[inputs["Family_Resilience"]],
    "Income_Group": income_map[inputs["Income_Group"]],
}

# Binary fields
binary_fields = [
    "Speech_Concern", "Interaction_Concern", "WordPhrase_Concern", "Behavior_Concern",
    "Says_One_Word", "Says_Two_Words", "Asks_Questions", "Breastfed", "Allergies",
    "Seizures", "Heart_Condition", "Asthma", "Premature_Birth", "Low_Birthweight",
    "Housing_Instability", "Food_Insecurity", "ADHD_Diagnosis", "Anxiety_Diagnosis",
    "Dev_Delay_Diagnosis"
]

for field in binary_fields:
    encoded[field] = encode_binary(inputs[field])

# Add continuous features
encoded["Age"] = inputs["Age"]
encoded["Family_Size"] = inputs["Family_Size"]
encoded["Adverse_Childhood_Exp"] = inputs["Adverse_Childhood_Exp"]
encoded["Screen_Time_Hours"] = inputs["Screen_Time_Hours"]
encoded["Sleep_Hours"] = inputs["Sleep_Hours"]

# Ensure input matches model feature order
for col in feature_columns:
    if col not in encoded:
        encoded[col] = 0  # default for missing fields

input_df = pd.DataFrame([encoded])[feature_columns]

# Predict
if st.button("🔍 Predict Autism Likelihood"):
    try:
        scaled = scaler.transform(input_df)
        prediction = model.predict(scaled)[0]
        proba = model.predict_proba(scaled)[0][1]
        if prediction == 1:
            st.error(f"🧩 Likelihood of autism (Confidence: {proba:.2%})")
        else:
            st.success(f"✅ Likelihood of autism (Confidence: {1 - proba:.2%})")
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
