
# Stroke Prediction App

This repository contains a Streamlit web application that predicts a patient's stroke risk using a saved stacked classifier and preprocessing artifacts. The app will attempt to load pre-trained artifacts on startup; if they are missing and the optional training dependency is installed, it can train and save a new model from the included CSV dataset.

## What’s in this repo
- `app.py` — Main Streamlit application. Loads pre-trained artifacts or (optionally) trains a new model.
- `healthcare-dataset-stroke-data.csv` — Dataset used for training (included in repo root).
- `stroke_model.joblib` — Serialized stacked model (if present).
- `scaler.joblib` — Saved StandardScaler for continuous features.
- `model_columns.joblib` — Saved column ordering used for aligning inputs at prediction time.
- `minorpro.ipynb` — Notebook used during development/EDA (optional).
- `requirements.txt` — Python dependencies.

> Note: The app's `app.py` expects the artifacts above (`stroke_model.joblib`, `scaler.joblib`, `model_columns.joblib`). If those are missing the app will try to train a new stacked classifier but training requires the `layerlearn` package (optional).

## Quick start (Windows — cmd.exe)

1. Create and activate a virtual environment

   python -m venv .venv
   .venv\Scripts\activate

2. Install dependencies

   pip install -r requirements.txt

   If you want to enable training from `app.py` (optional), ensure the `layerlearn` package is installed as well:

   pip install layerlearn

3. Ensure the dataset is present

   The app expects `healthcare-dataset-stroke-data.csv` in the repository root if training is required. If you only want to run predictions using the provided artifacts, no dataset is required.

4. Run the Streamlit app

   streamlit run app.py

   - On first run the app will try to load `stroke_model.joblib`, `scaler.joblib`, and `model_columns.joblib` from the project root.
   - If artifacts are missing and `layerlearn` is installed, the app will attempt to train a new stacked classifier and save the three artifacts.

## Files and artifacts explained

- `stroke_model.joblib` — The fitted FlexibleStackedClassifier (if present).
- `scaler.joblib` — StandardScaler fitted on continuous columns (`age`, `avg_glucose_level`, `bmi`).
- `model_columns.joblib` — Column order (after one-hot encoding) used at training time. The app aligns user input to this ordering before prediction.

If you want to force retraining, delete the three artifact files above (or run the training function directly) and restart the app with `layerlearn` available.

## How the app processes data (brief)

- Categorical features (one-hot encoded): `gender`, `work_type`, `smoking_status`, `ever_married`, `Residence_type`.
- Continuous features (scaled): `age`, `avg_glucose_level`, `bmi` (scaled using `scaler.joblib`).
- Binary features passed as 0/1: `hypertension`, `heart_disease`.

At prediction time the app:
- collects user inputs via the sidebar,
- one-hot encodes the categorical fields,
- reindexes columns to match `model_columns.joblib` (filling missing columns with zeros),
- applies the saved `scaler.joblib` to continuous columns,
- and uses `stroke_model.joblib` to predict class and probability.

## Retraining (optional)

To retrain inside the app you must have `layerlearn` installed (the training code uses `FlexibleStackedClassifier` from that package). Steps:

1. Install `layerlearn` in your environment.
2. Make sure `healthcare-dataset-stroke-data.csv` is in the repo root.
3. Start the app; when it cannot find saved artifacts it will attempt to train and save `stroke_model.joblib`, `scaler.joblib`, and `model_columns.joblib`.

Training can take several minutes depending on your machine.

## Troubleshooting

- If the app cannot load models: ensure `stroke_model.joblib`, `scaler.joblib`, and `model_columns.joblib` exist in the project root or install `layerlearn` and provide the CSV to allow training.
- If continuous columns are missing during prediction: make sure `model_columns.joblib` matches the encoder used at training time; retraining will regenerate it.
- If you see dependency errors: re-run `pip install -r requirements.txt` and install `layerlearn` if you need training capability.

## Disclaimer

This project is for demonstration and educational use only. It is not medical advice. The model outputs should not be used as a substitute for professional medical evaluation.


