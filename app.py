import pandas as pd
import numpy as np
import streamlit as st
import pickle
import matplotlib.pyplot as plt


with open('house_rent_model.pkl', 'rb') as file:
    model, scaler, label_encoders = pickle.load(file)


st.title("Pune House Rent Price Predictor")


bedroom = st.number_input("Number of Bedrooms", 1, 10, 2)
bathrooms = st.number_input("Number of Bathrooms", 1, 10, 2)
area = st.number_input("Area (in sqft)", 200, 10000, 500)
furnishing = st.selectbox("Furnishing", list(label_encoders['furnishing'].classes_))
avalable_for = st.selectbox("Available For", list(label_encoders['avalable_for'].classes_))
address = st.selectbox("Address", list(label_encoders['address'].classes_))
floor_number = st.number_input("Floor Number", 0, 50, 1)
facing = st.selectbox("Facing", list(label_encoders['facing'].classes_))
parking = st.number_input("Parking Spaces", 0, 10, 1)
deposit_amt = st.number_input("Deposit Amount (in Rs)", 0, 1000000, 50000)


def predict_rent(bedroom, bathrooms, area, furnishing, avalable_for, address, floor_number, facing, parking, deposit_amt):
    input_data = pd.DataFrame({
        'bedroom': [bedroom],
        'bathrooms': [bathrooms],
        'area': [area],
        'furnishing': [furnishing],
        'avalable_for': [avalable_for],
        'address': [address],
        'floor_number': [floor_number],
        'facing': [facing],
        'parking': [parking],
        'deposit_amt': [deposit_amt]
    })

    for column in input_data.columns:
        if column in label_encoders:
            input_data[column] = label_encoders[column].transform(input_data[column])

    input_data[['area', 'deposit_amt']] = scaler.transform(input_data[['area', 'deposit_amt']])
    rent_prediction = model.predict(input_data)
    return rent_prediction[0]


if st.button("Predict Rent"):
    predicted_rent = predict_rent(bedroom, bathrooms, area, furnishing, avalable_for, address, floor_number, facing, parking, deposit_amt)
    st.write(f"Predicted Rent: Rs {predicted_rent}")

