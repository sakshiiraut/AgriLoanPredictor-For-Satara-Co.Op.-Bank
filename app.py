import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
import joblib

@st.cache_data
def load_data():
    df = pd.read_csv("MLminiprojectdata - Sheet1.csv")
    
# Handling missing values 
    df.fillna(df.mean(numeric_only=True), inplace=True)
    
# Converting categorical to numerical
    df['Irrigation_Available'] = df['Irrigation_Available'].map({'Yes': 1, 'No': 0})
    df['Crop_Type'] = df['Crop_Type'].map({'Seasonal': 0, 'Perennial': 1})
    df['Subsidy_Access'] = df['Subsidy_Access'].map({'Yes': 1, 'No': 0})
    
    return df

df = load_data()

# Features and target

X = df[['Loan Period In Months', 'Age', 'Land_Acres', 'Irrigation_Available', 
        'Annual_Farming_Income', 'Farming_Equipment_Count', 'Crop_Type', 
        'Soil_Quality', 'Subsidy_Access']]

y = df['Sanctioned Amount']

# Standardization

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split 

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

#UI
st.title("Loan Customer Analysis")
st.subheader("Satara District Central CO-OP Bank LTD., Satara")

st.write("Enter the details below to predict the sanctioned loan amount:")

loan_period = st.number_input("Loan Period (in months)", min_value=1, step=1)
age = st.number_input("Age", min_value=18, max_value=100, step=1)

land_area_gunthe = st.number_input("Land Area (in Gunthe)", min_value=0.0, step=0.1)

# gunthe to acre 

if land_area_gunthe > 0:
    land_area_acre = land_area_gunthe / 40
    st.info(f"Converted Land Area: {land_area_acre:.2f} Acres")
irrigation_available = st.selectbox("Irrigation Available?", ["Yes", "No"])
annual_income = st.number_input("Annual Farming Income (₹)", min_value=0, step=1000)
equipment_count = st.number_input("Farming Equipment Count", min_value=0, step=1)
crop_type = st.selectbox("Crop Type", ["Seasonal", "Perennial"])
soil_quality = st.slider("Soil Quality (1=Poor, 10=Excellent)", min_value=1, max_value=10, step=1)
subsidy_access = st.selectbox("Subsidy Access?", ["Yes", "No"])

model_choice = st.selectbox("Select Model", ["Linear Regression", "Polynomial Regression", "Ridge Regression", "Lasso Regression"])

if st.button("Predict Loan Amount"):
    # Preprocess the input
    input_data = np.array([
        loan_period,
        age,
         land_area_acre, 
        1 if irrigation_available == "Yes" else 0,
        annual_income,
        equipment_count,
        0 if crop_type == "Seasonal" else 1,
        soil_quality,
        1 if subsidy_access == "Yes" else 0
    ]).reshape(1, -1)
    
    input_scaled = scaler.transform(input_data)
    
    # Model selection and prediction
    if model_choice == "Linear Regression":
        model = LinearRegression()
    elif model_choice == "Polynomial Regression":
        model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
    elif model_choice == "Ridge Regression":
        model = Ridge(alpha=1.0)
    elif model_choice == "Lasso Regression":
        model = Lasso(alpha=0.1)
    
    model.fit(X_train, y_train)
    prediction = model.predict(input_scaled)[0]
    
    prediction = max(prediction, 0)
    st.success(f"Predicted Sanctioned Loan Amount: ₹{prediction:,.2f}")



