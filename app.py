import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Load and train model
@st.cache_data
def load_and_train_model():
    data = pd.read_csv(r'C:\Users\HP\Downloads\archive\worldometer_data.csv')
    data = data.dropna(subset=['TotalCases', 'TotalDeaths', 'Population', 'TotalTests'])
    data['DeathRate'] = (data['TotalDeaths'] / data['TotalCases']) * 100
    data['HighDeathRate'] = (data['DeathRate'] > 3).astype(int)
    
    features = ['Population', 'TotalTests', 'Tests/1M pop', 'Tot Cases/1M pop']
    X = data[features]
    y = data['HighDeathRate']
    
    model = DecisionTreeClassifier(random_state=42, max_depth=5)
    model.fit(X, y)
    
    return model

# Title
st.title("🦠 COVID-19 Death Rate Predictor")
st.write("Predict if a country will have HIGH (>3%) or LOW (≤3%) death rate")

# Load model
model = load_and_train_model()

# User inputs
st.header("Enter Country Data:")

population = st.number_input("Population", min_value=1000, value=50000000, step=1000000)
total_tests = st.number_input("Total Tests Done", min_value=1000, value=5000000, step=100000)
tests_per_million = st.number_input("Tests per 1M Population", min_value=1000, value=100000, step=1000)
cases_per_million = st.number_input("Cases per 1M Population", min_value=100, value=10000, step=100)

# Predict button
if st.button("🔮 Predict Death Rate"):
    # Prepare input
    input_data = [[population, total_tests, tests_per_million, cases_per_million]]
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    
    # Show result
    if prediction == 1:
        st.error("⚠️ Prediction: HIGH Death Rate (>3%)")
        st.write("This country is predicted to have a death rate above 3%")
    else:
        st.success("✅ Prediction: LOW Death Rate (≤3%)")
        st.write("This country is predicted to have a death rate below 3%")