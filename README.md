# asd early risk screening app

Predict early risk of autism spectrum disorder using a supervised model trained on the 2023 National Survey of Children's Health (NSCH) dataset. This project combines rigorous data science—feature selection, class‑imbalance handling, model evaluation—with careful ML engineering, packaging the model into a reproducible Streamlit application.

## data science

- Extracted and cleaned 36 predictors from the NSCH dataset covering family, health, developmental and environmental factors.
- Balanced the minority class with SMOTE and tuned hyperparameters to optimize the Random Forest classifier.
- Evaluated performance using cross‑validation, precision–recall curves, and explained model behaviour with SHAP.

## ml engineering

- Built a Streamlit web app that serves the trained model with real‑time predictions and visual explanations.
- Encapsulated preprocessing (scaling, one‑hot encoding) and model inference in reusable scripts (`autism_predictor_app.py`) and pickled artefacts (`autism_random_forest_model.pkl`, `autism_scaler.pkl`).
- Provided a GitHub Actions workflow template for automated testing, linting, and container builds.

## repository structure

| path | description |
| --- | --- |
| `autism_predictor_app.py` | Streamlit application exposing the model and explanation plots |
| `autism_random_forest_model.pkl` | Serialized Random Forest classifier |
| `autism_scaler.pkl` | Fitted scaler used for input normalization |
| `feature_columns.json` | JSON file listing the exact set of feature names expected by the model |
| `categorical_distributions.json` | Categorical distribution values used for default inputs |
| `data/` | Contains the NSCH codebook and reference data |
| `requirements.txt` | Python dependencies |

## running the app locally

Clone the repo and install dependencies:

```bash
git clone https://github.com/Fam12345/asd‑early‑risk‑screening‑app
cd asd‑early‑risk‑screening‑app
pip install -r requirements.txt
streamlit run autism_predictor_app.py
```

A web page will open where you can fill out survey responses and view the predicted risk score along with model explanations.

---

This tool is for educational research purposes only; it is *not* a diagnostic instrument. Please consult healthcare professionals for clinical assessment.
