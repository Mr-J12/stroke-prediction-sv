 
# Stroke Prediction App

This repository provides a Streamlit web application that predicts a patient's stroke risk using a scikit-learn pipeline (preprocessing + stacking classifier). The app either loads a pre-trained pipeline (`stroke_pipeline.joblib`) or trains a new one from the provided CSV dataset on first run.

## What’s included
- `app.py` — Main Streamlit application (loads or trains the pipeline and serves the UI).
- `healthcare-dataset-stroke-data.csv` — Expected dataset used to train the model (CSV in repo root).
- `stroke_pipeline.joblib` — Serialized scikit-learn Pipeline produced by training (created after training completes).
- `minorpro.ipynb` — Notebook used during development/EDA (optional).
- `requirements.txt` — (present) Python dependencies; use it to install required packages.

## Quick start (Windows — cmd.exe)

1. Create and activate a virtual environment

   python -m venv .venv
   .venv\Scripts\activate

2. Install dependencies

   pip install -r requirements.txt

   If you prefer installing manually:
   pip install streamlit pandas scikit-learn joblib

3. Ensure the dataset `healthcare-dataset-stroke-data.csv` is in the project root. The app expects this filename by default. If your file is named differently, update the `RAW_DATA_PATH` constant in `app.py`.

4. Run the Streamlit app

   streamlit run app.py

   Note: On first run, if `stroke_pipeline.joblib` is missing the app will train the pipeline — this may take several minutes depending on your machine.

## Dataset format

The app expects a CSV with columns that map to the following internal feature names used by the pipeline:

id, gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type, avg_glucose_level, bmi, smoking_status, stroke

Example row (from the dataset):

33943,Female,39,0,0,Yes,Private,Urban,83.24,26.3,never smoked,1

- `stroke` is a binary target where 1 indicates stroke. During training the app normalizes column names and derives the `stroke` target if the CSV uses a different target column name.

If your CSV uses different headings, `app.py` contains a `rename_map` in `train_and_save_model()` that maps common alternative column names to the internal names — update that mapping if needed.

## How the model works

- Preprocessing
  - Continuous features: `age`, `avg_glucose_level`, `bmi` are scaled with `StandardScaler`.
  - Categorical features: `gender`, `work_type`, `smoking_status`, `ever_married`, `Residence_type` are one-hot encoded (`OneHotEncoder(handle_unknown='ignore', drop='first')`).
  - Binary passthrough: `hypertension`, `heart_disease` are passed through.

- Model
  - A stacking classifier is used: LogisticRegression as the base estimator and RandomForestClassifier as the meta-estimator.
  - The entire preprocessor + model is assembled into a scikit-learn `Pipeline` and saved to `stroke_pipeline.joblib`.

## Inputs and outputs

- Inputs: The app UI collects patient data (gender, age, hypertension, heart disease, marital status, work type, residence type, average glucose, BMI, smoking status).
- Outputs: Predicted class (0 = No Stroke, 1 = Stroke) and probability of stroke (displayed as a percentage). The app also provides tailored precautionary guidance based on probability ranges.

## Retraining and artifacts

- To force retraining, delete `stroke_pipeline.joblib` from the project root and restart the app. The app will train a new pipeline using the CSV dataset.
- The training function performs basic cleaning (numeric casting, filling `bmi` missing values with the mean, normalizing marital status strings, dropping single 'Other' gender rows if present) before fitting the pipeline.

## Troubleshooting

- "Raw data file not found": Place `healthcare-dataset-stroke-data.csv` in the repository root or change `RAW_DATA_PATH` in `app.py`.
- Unexpected categories in categorical columns: Re-train so the encoder learns the categories, or ensure `handle_unknown='ignore'` is enabled (the current pipeline uses this).
- Missing Python packages: Install packages with `pip install -r requirements.txt` or the manual install line above.

## Development notes

- The helper function `get_precautions(prob)` in `app.py` returns user-facing guidance based on the predicted probability. You can edit these messages to match local clinical guidance or language preferences.
- If you plan to deploy this app in production or share with clinicians, please consult domain experts and validate with proper clinical datasets.

## Security & medical disclaimer

This project is for demonstration and educational use only. It is not medical advice. The model outputs should not be used as a substitute for professional medical evaluation.


