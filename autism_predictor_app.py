import streamlit as st
import pandas as pd
import joblib
import json
# import plotly.express as px # You don't have this in your original, but needed for interactive version

# --- Load Model and Data (with caching for performance and error handling) ---
# Caching helps prevent reloading these large objects on every user interaction
@st.cache_resource
def load_model_and_scaler():
    """Loads the trained Random Forest model and scaler."""
    try:
        model = joblib.load("autism_random_forest_model.pkl")
        scaler = joblib.load("autism_scaler.pkl")
        return model, scaler
    except FileNotFoundError:
        st.error("Error: Model or scaler files not found. Please ensure 'autism_random_forest_model.pkl' and 'autism_scaler.pkl' are in the same directory.")
        st.stop() # Stop the app if essential files are missing early
    except Exception as e:
        st.error(f"Error loading model or scaler: {e}. This might be due to a scikit-learn version mismatch. Please check your dependencies.")
        st.stop() # Stop the app if essential files are corrupted or unreadable

@st.cache_data
def load_feature_columns():
    """Loads the list of feature columns used during training."""
    try:
        with open("feature_columns.json") as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("Error: Feature columns file not found. Please ensure 'feature_columns.json' is in the same directory.")
        st.stop() # Stop the app if essential files are missing early
    except Exception as e:
        st.error(f"Error loading feature columns: {e}. Check 'feature_columns.json' file integrity.")
        st.stop() # Stop the app if essential files are corrupted or unreadable


model, scaler = load_model_and_scaler()
feature_columns = load_feature_columns()

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
    🩺 **Note:** This tool is built using features derived from the **2023 National Survey of Children’s Health (NSCH)** and remains in a **preliminary research phase**.

    If you are interested in this work or would like to support the research,  
    please search for **Rabia Endris** on LinkedIn.
    """,
    unsafe_allow_html=True
)

# --- Initialize session state for storing prediction history ---
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []


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

# Handle conditional input for Low_Birthweight
if inputs["Premature_Birth"] == "Yes":
    inputs["Low_Birthweight"] = st.selectbox("Low birthweight?", ["Yes", "No"])
else:
    inputs["Low_Birthweight"] = "No" # Default to 'No' if not premature

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
# Use .get() with default 0 to handle potential missing features gracefully
input_data_for_prediction = {}
for col in feature_columns:
    input_data_for_prediction[col] = encoded.get(col, 0)

# Create DataFrame from the prepared dictionary
input_df = pd.DataFrame([input_data_for_prediction])[feature_columns]

# Predict
if st.button("🔍 Predict Autism Likelihood"):
    with st.spinner("Analyzing inputs and predicting likelihood..."): # Add spinner
        try:
            # Important: Convert scaled array back to DataFrame with feature names
            # to avoid 'X does not have valid feature names' warning
            scaled_array = scaler.transform(input_df)
            scaled_input_df = pd.DataFrame(scaled_array, columns=feature_columns)


            prediction = model.predict(scaled_input_df)[0]
            proba = model.predict_proba(scaled_input_df)[0] # Get both probabilities

            st.markdown("---") # Visual separator

            if prediction == 1:
                st.error(f"## 🧩 High Likelihood of Autism (Confidence: {proba[1]:.2%})")
                st.markdown("It is **highly recommended** to consult with a medical professional (e.g., pediatrician, developmental specialist, child psychologist) for a comprehensive evaluation.")
                st.info("Early intervention can make a significant difference. Consider reaching out to local autism support organizations for resources and guidance.")
                st.balloons() # Added visual effect for a significant finding
            else:
                st.success(f"## ✅ Low Likelihood of Autism (Confidence: {proba[0]:.2%})") # Use proba[0] for 'no autism' confidence
                st.markdown("Based on the provided information, the model indicates a **low likelihood** of autism.")
                st.info("If you still have concerns, please discuss them with your child's pediatrician during their next routine visit.")
                st.snow() # Added visual effect for positive news

            # --- Prediction History ---
            st.session_state.prediction_history.append({
                "Timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Age": inputs["Age"],
                "Sex": inputs["Sex"],
                "Race": inputs["Race"],
                "Prediction": "High Likelihood" if prediction == 1 else "Low Likelihood",
                "Confidence": f"{proba[1]:.2%}" if prediction == 1 else f"{proba[0]:.2%}"
            })

        except Exception as e:
            st.error(f"Prediction failed: {str(e)}. This might be due to a model loading issue or input processing error.")
            st.exception(e) # Display full traceback for debugging

# --- Optional: Display prediction history ---
st.markdown("---")
if st.session_state.prediction_history:
    with st.expander("View Previous Predictions"):
        history_df = pd.DataFrame(st.session_state.prediction_history)
        st.dataframe(history_df, use_container_width=True)
else:
    st.info("No previous predictions yet. Fill out the form and click 'Predict' to see history here.")