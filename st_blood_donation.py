import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model
model = pickle.load(open('logi_model.pkl', 'rb'))

# Streamlit app title
st.title("🩸 Blood Donation Prediction App")

# Input fields
months_since_last_donation = st.number_input("Months Since Last Donation", min_value=0)
number_of_donations = st.number_input("Number of Donations", min_value=0)
total_volume_donated = st.number_input("Total Volume Donated (in c.c)", min_value=0)
months_since_first_donation = st.number_input("Months Since First Donation", min_value=0)

# Predict button
if st.button("Predict Donation Probability"):
    # Prepare input for prediction
    input_data = np.array([[months_since_last_donation, number_of_donations,
                            total_volume_donated, months_since_first_donation]])
    
    # Make prediction
    probability = model.predict_proba(input_data)[:, 1][0]
    
    # Determine the result
    result = "Yes" if probability >= 0.5 else "No"
    
    # Display the result with probability
    st.write(f"### Will the person donate again? **{result}**")
    st.write(f"### Probability of donating again: **{probability:.2%}**")
    
    # Highlight based on result
    if result == "Yes":
        st.success("The person is likely to donate again.")
    else:
        st.error("The person is unlikely to donate again.")
