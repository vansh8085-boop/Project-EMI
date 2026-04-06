import streamlit as st
import joblib
import pandas as pd

# Load model & columns
model = joblib.load("best_model.pkl")
columns = joblib.load("columns.pkl")
target_encoder = joblib.load("target_encoder.pkl")

st.title("💰 EMI Risk Assessment")

st.header("Enter Customer Details")

# 🔹 Inputs (ALL REQUIRED)

age = st.number_input("Age", 18, 70)

gender = st.selectbox("Gender", ["Male", "Female"])
marital_status = st.selectbox("Marital Status", ["Single", "Married"])
education = st.selectbox("Education", ["High School", "Graduate", "Post Graduate", "Professional"])

monthly_salary = st.number_input("Monthly Salary")
employment_type = st.selectbox("Employment Type", ["Private", "Government", "Self-employed"])
years_of_employment = st.number_input("Years of Employment")
company_type = st.selectbox("Company Type", ["Small", "Medium", "Large"])

house_type = st.selectbox("House Type", ["Rented", "Own", "Family"])
monthly_rent = st.number_input("Monthly Rent")
family_size = st.number_input("Family Size")
dependents = st.number_input("Dependents")

school_fees = st.number_input("School Fees")
college_fees = st.number_input("College Fees")
travel_expenses = st.number_input("Travel Expenses")
groceries_utilities = st.number_input("Groceries & Utilities")
other_monthly_expenses = st.number_input("Other Monthly Expenses")

existing_loans = st.number_input("Existing Loans")
current_emi_amount = st.number_input("Current EMI Amount")

credit_score = st.number_input("Credit Score")
bank_balance = st.number_input("Bank Balance")
emergency_fund = st.number_input("Emergency Fund")

emi_scenario = st.selectbox("EMI Scenario", ["Shopping", "Appliances", "Vehicle", "Personal Loan", "Education"])
requested_amount = st.number_input("Requested Loan Amount")
requested_tenure = st.number_input("Requested Tenure")

# 🔹 Predict
if st.button("Predict"):

    # Create dictionary
    input_data = {
        "age": age,
        "gender": gender,
        "marital_status": marital_status,
        "education": education,
        "monthly_salary": monthly_salary,
        "employment_type": employment_type,
        "years_of_employment": years_of_employment,
        "company_type": company_type,
        "house_type": house_type,
        "monthly_rent": monthly_rent,
        "family_size": family_size,
        "dependents": dependents,
        "school_fees": school_fees,
        "college_fees": college_fees,
        "travel_expenses": travel_expenses,
        "groceries_utilities": groceries_utilities,
        "other_monthly_expenses": other_monthly_expenses,
        "existing_loans": existing_loans,
        "current_emi_amount": current_emi_amount,
        "credit_score": credit_score,
        "bank_balance": bank_balance,
        "emergency_fund": emergency_fund,
        "emi_scenario": emi_scenario,
        "requested_amount": requested_amount,
        "requested_tenure": requested_tenure
    }

    # Convert to dataframe
    input_df = pd.DataFrame([input_data])

    # 🔹 Convert to numeric (same as training)
    for col in input_df.columns:
        input_df[col] = input_df[col].astype(str)
        input_df[col] = input_df[col].replace(["nan", "None", ""], "0")
        input_df[col] = pd.to_numeric(input_df[col], errors="coerce")

    input_df.fillna(0, inplace=True)

    # 🔹 Ensure EXACT 30 columns
    input_df = input_df.reindex(columns=columns, fill_value=0)

    # 🔹 DEBUG (optional)
    st.write("Input shape:", input_df.shape)

    # Prediction
    pred = model.predict(input_df)

    # Convert back to label
    result = target_encoder.inverse_transform(pred)

    st.success(f"Prediction: {result[0]}")