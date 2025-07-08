import streamlit as st
import pandas as pd
import pickle
import os

# Function to load model
def load_model(model_name):
    model_path = os.path.join('models', model_name)
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

# Function to load the single classification report
def load_classification_report():
    report_path = os.path.join('reports', 'report.txt')
    with open(report_path, 'r') as file:
        report = file.read()
    return report

# ------------------- Streamlit App -------------------

st.set_page_config(page_title="Heart Failure Prediction", layout="centered")
st.title("❤️ Heart Failure Prediction Web App")

# Sidebar: Model selection
st.sidebar.header("🧠 Select Model")
model_names = [f for f in os.listdir('models') if f.endswith('.pkl')]
selected_model = st.sidebar.selectbox("Choose a model", model_names)
model = load_model(selected_model)
st.sidebar.success(f"Loaded model: {selected_model}")

# Tabs: Prediction | Report
tab1, tab2 = st.tabs(["🔍 Make Prediction", "📊 Classification Report"])

# ------------------- Tab 1: Prediction -------------------
with tab1:
    st.subheader("📝 Enter Patient Details")

    input_data = {
        'age': st.number_input("Age", min_value=0, max_value=120, value=45),
        'anaemia': st.selectbox("Anaemia", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes"),
        'creatinine_phosphokinase': st.number_input("Creatinine Phosphokinase", min_value=0, value=100),
        'diabetes': st.selectbox("Diabetes", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes"),
        'ejection_fraction': st.number_input("Ejection Fraction (%)", min_value=0, max_value=100, value=55),
        'high_blood_pressure': st.selectbox("High Blood Pressure", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes"),
        'platelets': st.number_input("Platelets (count)", min_value=0, value=250000),
        'serum_creatinine': st.number_input("Serum Creatinine", min_value=0.0, value=1.2),
        'serum_sodium': st.number_input("Serum Sodium", min_value=0.0, max_value=200.0, value=137.0),
        'sex': st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male"),
        'smoking': st.selectbox("Smoking", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes"),
        'time': st.number_input("Follow-up Period (in days)", min_value=0, value=100)

    }

    # Convert input to DataFrame
    input_df = pd.DataFrame([input_data])

    # Predict
    if st.button("🔮 Predict"):
        prediction = model.predict(input_df)[0]
        prediction_label = "❌ High Risk of Heart Failure" if prediction == 1 else "✅ Low Risk of Heart Failure"
        st.markdown(f"### Prediction: {prediction_label}")

# ------------------- Tab 2: Classification Report -------------------
with tab2:
    st.subheader("📋 Classification Report")
    report = load_classification_report()
    st.text(report)
