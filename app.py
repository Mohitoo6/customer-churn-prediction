import streamlit as st
import joblib
import numpy as np

# Load the saved model and scaler
model = joblib.load('random_forest_churn_model_compressed.pkl')
scaler = joblib.load('scaler.pkl')

# Streamlit App Configuration
st.set_page_config(
    page_title="CHURN-LENS",
    page_icon=":bar_chart:",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Add a Header with a Subheader
st.title("✨ CHURN-LENS: Customer Churn Prediction")
st.markdown(
    """
    **Welcome to ChurnLens**, a machine learning-powered app to predict customer churn.  
    Adjust the settings below, input customer details, and get actionable predictions!
    """
)

# Sidebar for Instructions
st.sidebar.header("🔧 How to Use")
st.sidebar.markdown(
    """
    1. Enter customer details in the form below.
    2. Adjust the **Churn Probability Threshold** as needed.
    3. Click on **Predict** to see the results.
    """
)

st.sidebar.info("Feel free to explore different thresholds to see how predictions change!")

# Main Input Form
with st.form("input_form"):
    st.subheader("📋 Enter Customer Details")
    credit_score = st.number_input("Credit Score", min_value=350, max_value=850, step=1, help="Customer's credit score.")
    age = st.number_input("Age", min_value=18, max_value=100, step=1, help="Customer's age in years.")
    tenure = st.number_input("Tenure (Months)", min_value=0, max_value=10, step=1, help="How long the customer has been with us (in months).")
    balance = st.number_input("Balance", min_value=0.0, step=0.01, help="Customer's current account balance.")
    num_of_products = st.selectbox("Number of Products", [1, 2, 3, 4], help="Number of products the customer has.")
    has_cr_card = st.selectbox("Has Credit Card", [1, 0], help="Does the customer have a credit card? (1 = Yes, 0 = No)")
    is_active_member = st.selectbox("Is Active Member", [1, 0], help="Is the customer an active member? (1 = Yes, 0 = No)")
    estimated_salary = st.number_input("Estimated Salary", min_value=0.0, step=0.01, help="Estimated annual salary of the customer.")
    geography = st.selectbox("Geography", ["France", "Germany", "Spain"], help="Customer's country of residence.")
    gender = st.selectbox("Gender", ["Female", "Male"], help="Customer's gender.")
    
    # Submit Button
    submitted = st.form_submit_button("Predict")

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

# Display Threshold Adjuster
st.subheader("⚙️ Adjust the Threshold")
threshold = st.slider(
    "Set Churn Probability Threshold", 
    0.0, 1.0, 0.5, step=0.01, help="Set the threshold for churn classification."
)

# Prediction Logic
if submitted:
    prediction_prob = model.predict_proba(scaled_input_data)[:, 1][0]
    
    # Display results with dynamic styling
    st.subheader("📊 Prediction Results")
    st.metric(label="Churn Probability", value=f"{prediction_prob:.2f}")
    st.metric(label="Threshold", value=f"{threshold:.2f}")
    
    if prediction_prob >= threshold:
        st.error(f"🚨 The customer is **likely to churn** with a probability of **{prediction_prob:.2f}**.")
    else:
        st.success(f"✅ The customer is **unlikely to churn** with a probability of **{1 - prediction_prob:.2f}**.")
    
    # Recommendations
    st.subheader("📌 Recommendations")
    if prediction_prob >= threshold:
        st.write("- Consider offering retention incentives.")
        st.write("- Focus on improving customer engagement.")
    else:
        st.write("- Maintain the current engagement strategy.")
