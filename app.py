import streamlit as st
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from layerlearn import FlexibleStackedClassifier


def train_and_save_model(csv_path="healthcare-dataset-stroke-data.csv"):
	if FlexibleStackedClassifier is None:
		raise ImportError(
			"layerlearn is required to train the stacked model. Install it or provide pre-trained joblib files."
		)

	# Load dataset
	try:
		data = pd.read_csv(csv_path)
	except FileNotFoundError:
		raise FileNotFoundError(f"CSV file not found at {csv_path}")

	# Preprocessing
	if 'id' in data.columns:
		data = data.drop('id', axis=1)

	# drop missing rows (consistent with original script)
	data = data.dropna()

	# fill missing bmi values with mean
	data.fillna(data.bmi.mean(), inplace=True)

	# remove single 'Other' gender if present
	if 'gender' in data.columns and 'Other' in data['gender'].values:
		gender_counts = data['gender'].value_counts()
		if gender_counts.get('Other', 0) == 1:
			data = data[data['gender'] != 'Other']

	X = data.drop('stroke', axis=1)
	y = data['stroke']

	categorical_cols = ['gender', 'work_type', 'smoking_status', 'ever_married', 'Residence_type']
	cols_to_encode = [col for col in categorical_cols if col in X.columns]

	# One-hot encoding (drop_first=True as in original train_model)
	X_processed = pd.get_dummies(X, columns=cols_to_encode, drop_first=True)

	continuous_cols = ['age', 'avg_glucose_level', 'bmi']
	scaler = StandardScaler()
	# Fit-transform continuous columns
	for col in continuous_cols:
		if col not in X_processed.columns:
			raise KeyError(f"Expected continuous column '{col}' not found in data after encoding.")

	X_processed[continuous_cols] = scaler.fit_transform(X_processed[continuous_cols])

	# Save column order
	model_columns = X_processed.columns
	joblib.dump(model_columns, 'model_columns.joblib')

	# Train/test split
	X_train, X_test, y_train, y_test = train_test_split(
		X_processed, y, test_size=0.3, random_state=42, stratify=y
	)

	# Define models
	base_model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
	meta_model = RandomForestClassifier(random_state=0, class_weight='balanced')

	fsc = FlexibleStackedClassifier(base_model, meta_model)
	fsc.fit(X_train, y_train)

	# Evaluation (prints)
	y_pred = fsc.predict(X_test)
	acc = accuracy_score(y_test, y_pred)
	report = classification_report(y_test, y_pred, target_names=['No Stroke', 'Stroke'])
	print(f"Model accuracy: {acc:.4f}")
	print(report)

	# Save artifacts
	joblib.dump(fsc, 'stroke_model.joblib')
	joblib.dump(scaler, 'scaler.joblib')
	print("Saved stroke_model.joblib, scaler.joblib, model_columns.joblib")

	return fsc, scaler, model_columns



# --- Load or train model assets ---
@st.cache_data
def load_or_train():
	# Try loading existing artifacts; otherwise train
	try:
		model = joblib.load('stroke_model.joblib')
		scaler = joblib.load('scaler.joblib')
		model_columns = joblib.load('model_columns.joblib')
		return model, scaler, model_columns
	except Exception:
		# Train the model (may raise if dependencies missing)
		model, scaler, model_columns = train_and_save_model()
		return model, scaler, model_columns


def main():
	st.set_page_config(page_title="Stroke Prediction App", page_icon="🧠", layout="wide")

	st.title("🧠 Stroke Prediction Model (Combined)")
	st.markdown("This single file will train the model if necessary and then show the Streamlit UI.")

	try:
		model, scaler, model_columns = load_or_train()
	except Exception as e:
		st.error(f"Failed to load or train model: {e}")
		st.stop()

	# Sidebar inputs
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
				input_df = input_df_raw.copy()
				input_df_processed = pd.get_dummies(input_df)

				# Align columns with training columns
				input_df_aligned = input_df_processed.reindex(columns=model_columns, fill_value=0)

				# Apply scaler to continuous columns if present
				continuous_cols = ['age', 'avg_glucose_level', 'bmi']
				cols_present = [c for c in continuous_cols if c in input_df_aligned.columns]
				if len(cols_present) == len(continuous_cols):
					input_df_aligned[continuous_cols] = scaler.transform(input_df_aligned[continuous_cols])

				prediction = model.predict(input_df_aligned)[0]
				probability = model.predict_proba(input_df_aligned)[0]
				prob_stroke = probability[1]

				st.markdown("---")
				st.subheader("Prediction Result")
				col1, col2 = st.columns(2)
				if prediction == 1:
					col1.error("Prediction: **STROKE**")
					col1.markdown("The model predicts a **high risk** of stroke.")
				else:
					col1.success("Prediction: **NO STROKE**")
					col1.markdown("The model predicts a **low risk** of stroke.")

				col2.metric(
					label="Probability of Stroke",
					value=f"{prob_stroke * 100:.2f}%",
					delta=f"{prob_stroke * 100 - 5:.2f}% vs 5% average",
				)
				st.progress(prob_stroke)

				# --- Precautionary guidance based on probability ---
				def get_precautions(prob):
					"""Return a markdown-formatted precaution message based on probability (0-1).
			
					The messages give progressively stronger advice and clear next steps, including when to seek urgent care,
					screening tests to ask about, medication/monitoring notes, lifestyle specifics, and suggested follow-up timing.
					"""
					p = prob * 100
					# Core, reusable lifestyle guidance used in multiple bands
					common_include = (
						"- Diet: emphasize whole grains, vegetables, fruit, legumes, lean protein (fish, poultry), and nuts.\n"
						"- Fats: prefer olive oil and oily fish (omega-3); limit saturated fats and avoid trans fats.\n"
						"- Salt & sugar: reduce added salt and sugary drinks; prefer fresh foods over processed.\n"
						"- Activity & weight: aim for ~150 minutes of moderate activity per week and maintain a healthy BMI.\n"
						"- Tobacco & alcohol: stop smoking; limit alcohol to guideline amounts or avoid if advised.\n"
					)

					# When to seek immediate care (applies across higher-risk bands)
					urgent_signs = (
						"**If you develop any of these sudden symptoms, seek emergency care immediately:**\n"
						"- Sudden weakness or numbness of the face, arm or leg (especially on one side)\n"
						"- Sudden confusion, trouble speaking or understanding speech\n"
						"- Sudden trouble seeing in one or both eyes\n"
						"- Sudden trouble walking, dizziness, loss of balance or coordination\n"
						"- Sudden severe headache with no known cause\n\n"
					)

					if p < 10:
						return (
							"### Precautions — Low risk (0–10%)\n"
							"Your predicted risk is low. Continue healthy habits and routine preventive care.\n\n"
							"**Daily & lifestyle:**\n"
							f"{common_include}\n\n"
							"**Medical & monitoring:**\n"
							"- Routine checks: measure blood pressure at least annually and follow your clinician's screening schedule for cholesterol and diabetes.\n"
							"- If you have known conditions (hypertension, diabetes, AF), follow your treatment plan and appointments.\n\n"
							"**Follow-up:**" " Annual general review; sooner if symptoms or new risk factors arise."
						)
					elif p < 20:
						return (
							"### Precautions — Mild risk (10–20%)\n"
							"Mildly elevated risk — small, consistent changes can lower your long-term risk.\n\n"
							"**Lifestyle actions (practical):**\n"
							f"{common_include}\n"
							"- Start by tracking food intake and aim to reduce processed foods 3–5 days/week.\n"
							"- Add two short (10–20 min) brisk walks per day if currently inactive.\n\n"
							"**Medical & screening:**\n"
							"- Check blood pressure, fasting lipids, and HbA1c (if overweight or family history of diabetes).\n"
							"- Discuss risk-lowering strategies with your GP (dietary counselling, smoking cessation support).\n\n"
							"**Follow-up:**" " Primary-care review within 3 months to set targets and review progress."
						)
					elif p < 30:
						return (
							"### Precautions — Moderate risk (20–30%)\n"
							"Moderate probability — active measures and medical review are recommended.\n\n"
							"**Immediate lifestyle priorities:**\n"
							f"{common_include}\n"
							"- Consider a structured diet plan (DASH or Mediterranean-style) and a progressive exercise program.\n\n"
							"**Medical & investigations to ask about:**\n"
							"- Blood pressure monitoring (home or ambulatory), fasting lipids, HbA1c, and kidney function.\n"
							"- Discuss aspirin or statins only as advised by your clinician — do not start them without medical guidance.\n"
							"- If palpitations or irregular pulse, request an ECG to check for atrial fibrillation.\n\n"
							"**Follow-up & monitoring:**" " GP or clinic review within 1 month to set targets and repeat tests as needed."
						)
					elif p < 50:
						return (
							"### Precautions — High risk (30–50%)\n"
							"High predicted risk — prompt medical evaluation and a clear prevention plan are advised.\n\n"
							"**Priority medical actions:**\n"
							"- Urgently review blood pressure control; many strokes are prevented by good BP management.\n"
							"- Full cardiovascular risk workup: lipids, HbA1c, kidney function, ECG, and possibly imaging or referral as recommended.\n"
							"- Review current medications (antihypertensives, statins, anticoagulants if indicated) and adherence.\n\n"
							"**Lifestyle & support:**\n"
							f"{common_include}\n"
							"- Consider referral to a structured risk-reduction program (nutritionist, smoking cessation, supervised exercise).\n\n"
							f"{urgent_signs}"
							"**Follow-up:**" " Specialist or primary-care follow-up within 1–2 weeks depending on findings."
						)
					elif p < 70:
						return (
							"### Precautions — Very high risk (50–70%)\n"
							"This is a very high predicted probability — seek prompt medical assessment and act quickly to reduce risk.\n\n"
							"**Immediate steps (do these now):**\n"
							"- Arrange an urgent appointment with your GP or local urgent care to review blood pressure and medications.\n"
							"- If you have not had recent heart rhythm monitoring, ask about an ECG or ambulatory monitor.\n\n"
							"**Medical & likely interventions:**\n"
							"- Rapid optimisation of blood pressure and cholesterol—this may include starting or adjusting medications.\n"
							"- Consider specialist referral (cardiology or stroke prevention clinic) for multi-factorial assessment.\n\n"
							f"{urgent_signs}"
							"**Follow-up:**" " Clinical review within days and close monitoring until risk factors are controlled."
						)
					else:
						return (
							"### Precautions — Critical / highest risk (70–100%)\n"
							"Predicted probability is very high — this indicates an urgent need for medical evaluation and likely rapid intervention.\n\n"
							"**Do not delay:**\n"
							f"{urgent_signs}"
							"**Immediate actions:**\n"
							"- Contact emergency services or attend an emergency department if symptomatic or if you cannot get rapid primary care assessment.\n"
							"- If asymptomatic but high risk, arrange same-day or next-day urgent review with primary care or a stroke prevention service.\n\n"
							"**Medical:**\n"
							"- Expect rapid investigations (bloods, ECG, imaging) and fast-tracked specialist input.\n"
							"- Management may include urgent optimisation of blood pressure, start/adjust statin therapy, anticoagulation if atrial fibrillation is found, and other targeted therapies as clinically indicated.\n\n"
							"**Support & planning:**\n"
							"- Arrange help at home if mobility or function is affected; involve family/carers in planning.\n"
							"**Follow-up:**" " Close specialist-led follow-up and monitoring; consider multidisciplinary rehabilitation or secondary prevention pathways."
						)

				precautions_md = get_precautions(prob_stroke)
				st.markdown(precautions_md)

			except Exception as e:
				st.error(f"An error occurred during prediction: {e}")
	else:
		st.info("Click the 'Predict Likelihood' button in the sidebar to see the result.")


if __name__ == "__main__":
	main()