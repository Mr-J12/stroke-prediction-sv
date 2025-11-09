import streamlit as st
import pandas as pd
import joblib
import warnings
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
warnings.filterwarnings('ignore')

# --- Constants ---
# Use the provided dataset with new column names
RAW_DATA_PATH = "healthcare-dataset-stroke-data.csv"
PIPELINE_PATH = "stroke_pipeline.joblib"

# Define feature groups for the pipeline
CONTINUOUS_COLS = ['age', 'avg_glucose_level', 'bmi']
PASSTHROUGH_COLS = ['hypertension', 'heart_disease']
CATEGORICAL_COLS = ['gender', 'work_type', 'smoking_status', 'ever_married', 'Residence_type']
TARGET_COL = 'stroke'


# --- 1. Model Training Function ---
def train_and_save_model():
    """
    Loads raw data, trains a full preprocessing and model pipeline,
    evaluates it, saves the artifact, and returns the pipeline.
    This function is called by the loader if the model file isn't found.
    """
    st.info(f"Loading data from {RAW_DATA_PATH}...")
    # Load dataset
    try:
        data = pd.read_csv(RAW_DATA_PATH)
    except FileNotFoundError:
        st.error(f"Error: Raw data file not found at {RAW_DATA_PATH}")
        st.error("Please download 'healthcare-dataset-stroke-data.csv' and place it in the same directory.")
        st.stop()
        return None

    # --- Data Cleaning ---
    # Rename columns from the provided CSV to the internal names expected by the pipeline
    rename_map = {
        'Age': 'age',
        'Gender': 'gender',
        'Hypertension': 'hypertension',
        'Heart Disease': 'heart_disease',
        'Marital Status': 'ever_married',
        'Work Type': 'work_type',
        'Residence Type': 'Residence_type',
        'Average Glucose Level': 'avg_glucose_level',
        'Body Mass Index (BMI)': 'bmi',
        'Smoking Status': 'smoking_status',
        'Diagnosis': 'diagnosis'
    }

    data = data.rename(columns=rename_map)

    # Drop columns that are not used as features
    data.drop(['id'], axis=1, inplace=True)

    # Normalize some columns and fill missing values
    if 'bmi' in data.columns:
        data['bmi'] = pd.to_numeric(data['bmi'], errors='coerce')
        data['bmi'] = data['bmi'].fillna(data['bmi'].mean())

    # Standardize 'ever_married' values to 'Yes'/'No'
    if 'ever_married' in data.columns:
        data['ever_married'] = data['ever_married'].astype(str).str.strip().replace(
            {'Married': 'Yes', 'Single': 'No', 'married': 'Yes', 'single': 'No'})

    # Standardize target column from 'diagnosis' -> 'stroke' (1 if exact 'Stroke')
    if 'diagnosis' in data.columns:
        data['stroke'] = data['diagnosis'].astype(str).str.strip().str.lower().eq('stroke').astype(int)
        data = data.drop(columns=['diagnosis'])

    # Remove single 'Other' gender if present (as before)
    if 'gender' in data.columns and 'Other' in data['gender'].values:
        gender_counts = data['gender'].value_counts()
        if gender_counts.get('Other', 0) == 1:
            data = data[data['gender'] != 'Other']
    
    st.info("Data cleaning complete. Defining pipeline...")

    # --- Preprocessing Pipeline ---
    continuous_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', continuous_transformer, CONTINUOUS_COLS),
            ('cat', categorical_transformer, CATEGORICAL_COLS),
            ('pass', 'passthrough', PASSTHROUGH_COLS)
        ],
        remainder='drop'
    )
    
    # --- Model Definition (Stacking) ---
    base_model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    meta_model = RandomForestClassifier(random_state=0, class_weight='balanced')

    estimators = [('lr', base_model)]
    
    stacking_model = StackingClassifier(
        estimators=estimators,
        final_estimator=meta_model,
        cv=5
    )

    # --- Create the Full Pipeline ---
    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', stacking_model)
    ])
    
    # --- Train and Evaluate ---
    X = data.drop(TARGET_COL, axis=1)
    y = data[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    st.info("Training model... This may take a moment.")
    full_pipeline.fit(X_train, y_train)
    st.info("Training complete.")

    # Evaluation (prints to console)
    y_pred = full_pipeline.predict(X_test)
    print("\n--- Model Evaluation (Printed to Console) ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred, target_names=['No Stroke', 'Stroke']))
    print("---------------------------------------------")

    # --- Save Artifact ---
    joblib.dump(full_pipeline, PIPELINE_PATH)
    st.success(f"Model trained and saved to {PIPELINE_PATH}")
    
    return full_pipeline


# --- 2. Model Loading Function ---
@st.cache_resource
def load_or_train_model():
    """
    Tries to load the pipeline from disk. If not found,
    it triggers the training process.
    """
    try:
        # Try loading existing model
        model = joblib.load(PIPELINE_PATH)
        st.info("Loaded pre-trained model from disk.")
        return model
    except FileNotFoundError:
        # If model not found, train it
        st.warning(f"No model found at {PIPELINE_PATH}. Training a new model...")
        with st.spinner("Training model. This will only happen once..."):
            model = train_and_save_model()
        return model
    except Exception as e:
        st.error(f"An error occurred while loading or training the model: {e}")
        st.stop()
        return None

# --- 3. UI Helper Function ---
def get_precautions(prob):
    """Return a markdown-formatted precaution message based on probability (0-1)."""
    p = prob * 100
    common_include = (
        "- **Diet:** Eat whole grains, fruits, vegetables, and lean protein. Avoid processed foods, sugary drinks, and excessive salt.\n"
        "- **Lifestyle:** Maintain a healthy weight, exercise 30 minutes most days, stop smoking, and limit alcohol."
    )
    if p < 10:
        return (
            "### Precautions — Low risk (0-10%)\n"
            "You are at low predicted risk. Keep up the healthy habits!\n\n"
            f"{common_include}\n\n"
            "**Medical:** Routine annual checkups."
        )
    elif p < 20:
        return (
            "### Precautions — Mild risk (10-20%)\n"
            "You have a mild predicted risk. Small, sustainable changes now can noticeably lower long-term risk.\n\n"
            "**Diet (examples):**\n"
            "- Aim for 3 servings of vegetables and 2 servings of fruit daily.\n"
            "- Swap refined grains for whole grains (oats, brown rice, whole-wheat bread).\n"
            "- Use olive oil instead of butter; include oily fish (e.g., salmon) twice weekly.\n\n"
            "**Exercise & lifestyle:**\n"
            "- Start with 20–30 minutes of moderate activity (brisk walk) most days.\n"
            "- Reduce sedentary time (stand/stretch hourly).\n\n"
            f"{common_include}\n\n"
            "**Monitoring & medical:**\n"
            "- Check blood pressure at home or clinic every 1–3 months.\n"
            "- Get a baseline lipid panel if you haven't in the last year.\n"
            "- Discuss any family history of cardiovascular disease with your GP."
        )
    elif p < 30:
        return (
            "### Precautions — Moderate risk (20-30%)\n"
            "Moderate risk indicates that targeted interventions are helpful now to reduce near-term and long-term stroke risk.\n\n"
            "**Diet (practical steps):**\n"
            "- Follow a Mediterranean-style pattern: lots of vegetables, legumes, whole grains, nuts, and olive oil.\n"
            "- Limit red and processed meats to small portions and few times per week.\n"
            "- Reduce added sugars and sugary beverages—prefer water, tea, or coffee without sugar.\n\n"
            "**Exercise & weight:**\n"
            "- Build up to 150 minutes/week of moderate aerobic activity (e.g., 30 min × 5 days).\n"
            "- Add two strength sessions weekly (bodyweight or light weights).\n\n"
            f"{common_include}\n\n"
            "**Monitoring & medical:**\n"
            "- Get blood pressure, fasting lipids, and HbA1c (if overweight or family history) within the next 1–3 months.\n"
            "- If BP is consistently ≥130/80 mmHg or LDL cholesterol is elevated, discuss prevention strategies with your clinician (diet, lifestyle, and possibly medication).\n"
            "- If you have atrial fibrillation, palpitations, or other cardiac symptoms, seek evaluation."
        )
    elif p < 50:
        return (
            "### Precautions — High risk (30-50%)\n"
            "High predicted risk — act now with both lifestyle changes and timely medical assessment to reduce risk.\n\n"
            "**Diet (action plan):**\n"
            "- Adopt a low-salt (≤5–6 g/day), low-saturated-fat diet. Replace snacks with nuts, fruit, or yogurt.\n"
            "- Prioritize fiber: legumes, whole grains, and vegetables at every meal.\n"
            "- If overweight, aim for gradual 5–10% weight loss over months (reduces BP and glucose).\n\n"
            "**Exercise & daily routine:**\n"
            "- Target 150–300 minutes/week of moderate aerobic activity, or 75–150 minutes/week vigorous activity, per tolerance.\n"
            "- Begin a supervised or guided exercise program if you have chronic conditions.\n\n"
            f"{common_include}\n\n"
            "**Monitoring & medical (urgent):**\n"
            "- Book a GP visit within weeks for full cardiovascular risk assessment (BP, lipids, glucose, BMI).\n"
            "- Expect a personalized plan: intensified monitoring, possible initiation of blood-pressure or lipid-lowering therapy, and lifestyle referral (dietitian or cardiac rehab).\n"
            "- If you experience transient neurological signs (sudden numbness, slurred speech, facial droop) seek emergency care immediately."
        )
    elif p < 60:
        return (
            "### Precautions — Very High risk (50-60%)\n"
            "This predicted probability is high — arrange prompt medical review and tighten preventive measures.\n\n"
            "**Diet & Lifestyle (urgent):**\n"
            "- Strictly reduce salt and saturated fat; increase vegetables, fruits, and fiber.\n"
            "- Avoid processed foods, trans fats, and sugary drinks.\n"
            "- Moderate daily activity as advised by a clinician; stop smoking and limit alcohol.\n\n"
            "**Medical (recommended):**\n"
            "- Schedule a timely GP appointment for cardiovascular risk assessment.\n"
            "- Get blood pressure, lipid profile, and blood sugar (HbA1c) checked.\n"
            "- Discuss starting or optimizing BP/cholesterol medications if indicated.\n"
            "- Plan closer follow-up (weeks to a few months) depending on clinician advice."
        )
    elif p < 70:
        return (
            "### Precautions — Severe risk (60-70%)\n"
            "This is a very high predicted probability — seek urgent medical assessment and consider expedited specialist referral.\n\n"
            "**Immediate actions:**\n"
            "- Avoid heavy physical exertion until medically cleared.\n"
            "- Ensure strict dietary restrictions: very low salt, low saturated fat; prioritize whole foods.\n"
            "- Prepare to act on any sudden neurological symptoms (weakness, numbness, speech trouble, severe headache).\n\n"
            "**Medical (urgent):**\n"
            "- Arrange an urgent GP or emergency visit depending on symptoms.\n"
            "- Consider expedited diagnostic workup (imaging, ECG, blood tests) and specialist referral (cardiology/neurology).\n"
            "- Review and potentially intensify medications for blood pressure, cholesterol, or anticoagulation as clinically indicated.\n"
            "- Close follow-up within days to a couple of weeks is recommended."
        )
    else:  # p >= 70
        return (
            "### Precautions — Critical/highest risk (70-100%)\n"
            "Predicted probability is very high. Seek immediate medical evaluation.\n\n"
            "**Do not delay:**\n"
            "- Contact a healthcare provider or emergency services if symptomatic.\n"
            "- Follow any current medication plans and do not stop critical medicines without advice.\n\n"
            "**Diet & lifestyle:** Strict medical diet (very low salt, very low saturated fat), stop smoking, avoid alcohol.\n\n"
            "**Medical:** Specialist referral, urgent imaging/tests (CT/MRI) and possible hospitalization/intervention as recommended by physicians."
        )


# --- 4. Main Application ---
def main():
    st.set_page_config(page_title="Stroke Prediction App", page_icon="🧠", layout="wide")

    st.title("🧠 Stroke Prediction Model")
    st.markdown("This app uses a machine learning pipeline to predict stroke risk. It will train the model on first run if no pre-trained file is found.")

    # Load the model. This will train it if not found.
    model = load_or_train_model()
    if model is None:
        # Loader function already showed the error
        st.stop()

    # --- Sidebar Inputs ---
    st.sidebar.header("Patient Input Features")
    st.sidebar.markdown("Please fill in the details of the patient.")

    def user_input_features():
        gender = st.sidebar.selectbox("Gender", ("Male", "Female"))
        age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=50, step=1)
        hypertension = st.sidebar.selectbox("Hypertension", ("No", "Yes"))
        heart_disease = st.sidebar.selectbox("Heart Disease", ("No", "Yes"))
        ever_married = st.sidebar.selectbox("Ever Married", ("Yes", "No"))
        work_type = st.sidebar.selectbox("Work Type", ("Private", "Self-employed", "Govt_job", "children", "Never_worked"))
        residence_type = st.sidebar.selectbox("Residence Type", ("Urban", "Rural"))
        avg_glucose_level = st.sidebar.number_input("Average Glucose Level", min_value=10.0, max_value=400.0, value=100.0, step=0.1)
        bmi = st.sidebar.number_input("BMI", min_value=5.0, max_value=100.0, value=28.0, step=0.1)
        smoking_status = st.sidebar.selectbox("Smoking Status", ("formerly smoked", "never smoked", "smokes", "Unknown"))

        data = {
            'gender': gender,
            'age': age,
            'hypertension': 1 if hypertension == "Yes" else 0,
            'heart_disease': 1 if heart_disease == "Yes" else 0,
            'ever_married': ever_married,
            'work_type': work_type,
            'Residence_type': residence_type,
            'avg_glucose_level': avg_glucose_level,
            'bmi': bmi,
            'smoking_status': smoking_status
        }
        return pd.DataFrame([data])

    input_df_raw = user_input_features()

    st.subheader("Patient's Input Data")
    st.write(input_df_raw)

    if st.sidebar.button("Predict Likelihood"):
        with st.spinner("Processing and predicting..."):
            try:
                # The pipeline handles ALL preprocessing
                prediction = model.predict(input_df_raw)[0]
                probability = model.predict_proba(input_df_raw)[0]
                prob_stroke = probability[1] # Probability of class 1 (Stroke)

                st.markdown("---")
                st.subheader("Prediction Result")
                col1, col2 = st.columns([1, 2])
                
                if prediction == 1:
                    col1.error("Prediction: **STROKE**")
                    col1.markdown("The model predicts a **high risk** of stroke.")
                else:
                    col1.success("Prediction: **NO STROKE**")
                    col1.markdown("The model predicts a **low risk** of stroke.")

                col2.metric(
                    label="Probability of Stroke",
                    value=f"{prob_stroke * 100:.2f}%"
                )
                st.progress(prob_stroke)

                # --- Precautionary guidance ---
                precautions_md = get_precautions(prob_stroke)
                st.markdown(precautions_md)
                
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
    else:
        st.info("Click the 'Predict Likelihood' button in the sidebar to see the result.")


if __name__ == "__main__":
    main()