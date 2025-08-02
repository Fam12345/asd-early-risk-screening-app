# 🧠 Autism Early Screening Tool

This is a **Streamlit-based web app** built to support awareness and early screening of autism-related behavioral and developmental traits in children. It uses a machine learning model trained on data from the **2023 National Survey of Children’s Health (NSCH)**, with 36 predictors drawn from family, health, developmental, and environmental indicators.

---

## 🧪 Research Status

This tool is part of a **preliminary research project**. It is **not a diagnostic tool**. The predictions it generates are for **educational and awareness** purposes only.

A pediatric nurse is featured in the app to remind users that **screening is not diagnosis**—professional evaluation is essential.

**If you're interested in this research or want to support it, find Rabia Endris on LinkedIn.**

---

## 📁 Repository Contents

- `autism_predictor_app.py` — the Streamlit application code
- `autism_random_forest_model.pkl` — trained Random Forest classifier
- `autism_scaler.pkl` — `StandardScaler` used for input normalization
- `feature_columns.json` — JSON file listing the exact 36 model input features
- `Fully_Categorized_Autism_Variables.csv` — categorical training inputs
- `CAHMI DRC 2023 NSCH SPSS Codebook.pdf` — original codebook from the national survey

---

## 🧮 Model Summary

- **Algorithm**: Random Forest (scikit-learn)
- **Inputs**: 36 categorical and numeric child/family characteristics
- **Output**: Binary classification (High/Low likelihood of autism) with confidence
- **Data Source**: 2023 National Survey of Children’s Health (NSCH)
- **Frameworks**: `streamlit`, `pandas`, `scikit-learn`, `joblib`

---

## 💻 Run the App Locally

### 1. Clone this repository
```bash
git clone https://github.com/Fam12345/Autism_Predictor.git
cd Autism_Predictor
