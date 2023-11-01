import streamlit as st
import pickle
import pandas as pd
from pycaret.classification import load_model

model = load_model('rf_pipeline') 

# Load the trained model

# with open('rf_pipeline', 'rb') as model_file:
#     model = pickle.load(model_file)
    # print(model)
print(type(model))
# Function to make predictions
def predict_program_status(spaSavingAmount, Total_loan):
    # Create a DataFrame with the user's input
    input_data = pd.DataFrame({'spaSavingAmount': [spaSavingAmount],
                               'Total_loan': [Total_loan],
                               })
    
    # Make predictions
    prediction = model.predict(input_data)
    
    return prediction[0]  # Assuming your model returns a single prediction

# Streamlit app
st.title('Debt Relief Program Predictor')

# User input
spaSavingAmount = st.number_input('Special Purpose Saving Amount', min_value=0)
Total_loan = st.number_input('Total Current Outstanding', min_value=0)
# creditorBalance = st.number_input('Creditor Balance', min_value=0)

if st.button('Predict'):
    # Make predictions
    prediction = predict_program_status(spaSavingAmount, Total_loan)
    
    # Display the prediction
    st.write(f'Predicted Program Status: {prediction}')

# Optionally, you can add more content and instructions to your Streamlit app.


