import streamlit as st
import joblib
import numpy as np

# Load the saved model and scaler
model = joblib.load('random_forest_churn_model_compressed.pkl')
scaler = joblib.load('scaler.pkl')

# Streamlit App
st.title("CHURN-LENS")
st.write("Enter customer details to predict churn probability:")

# Input fields for customer details
credit_score = st.number_input("Credit Score", min_value=350, max_value=850, step=1)
age = st.number_input("Age", min_value=18, max_value=100, step=1)
tenure = st.number_input("Tenure (Months)", min_value=0, max_value=10, step=1)
balance = st.number_input("Balance", min_value=0.0, step=0.01)
num_of_products = st.selectbox("Number of Products", [1, 2, 3, 4])
has_cr_card = st.selectbox("Has Credit Card (1 = Yes, 0 = No)", [1, 0])
is_active_member = st.selectbox("Is Active Member (1 = Yes, 0 = No)", [1, 0])
estimated_salary = st.number_input("Estimated Salary", min_value=0.0, step=0.01)
geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.selectbox("Gender", ["Female", "Male"])

# Convert categorical inputs to numerical
geography_germany = 1 if geography == "Germany" else 0
geography_spain = 1 if geography == "Spain" else 0
gender_male = 1 if gender == "Male" else 0

# Create input array
input_data = np.array([[credit_score, age, tenure, balance, num_of_products, 
                        has_cr_card, is_active_member, estimated_salary, 
                        geography_germany, geography_spain, gender_male]])

# Scale the input data
scaled_input_data = scaler.transform(input_data)

# Use session state to store results
if "prediction_prob" not in st.session_state:
    st.session_state.prediction_prob = None
    st.session_state.threshold = 0.5

# Threshold slider for dynamic adjustment
st.write("### Adjust the Threshold for Churn Classification")
threshold = st.slider("Set Churn Probability Threshold", 0.0, 1.0, 0.5, key="threshold")

# Predict churn probability
if st.button("Predict"):
    prediction_prob = model.predict_proba(scaled_input_data)[:, 1][0]
    st.session_state.prediction_prob = prediction_prob  # Save the result in session state

# Display results
if st.session_state.prediction_prob is not None:
    prediction_prob = st.session_state.prediction_prob
    st.write(f"### Prediction Results")
    st.write(f"- **Churn Probability**: {prediction_prob:.2f}")
    st.write(f"- **Current Threshold**: {st.session_state.threshold:.2f}")
    
    if prediction_prob >= st.session_state.threshold:
        st.error(f"The customer is likely to churn.")
    else:
        st.success(f"The customer is unlikely to churn.")
