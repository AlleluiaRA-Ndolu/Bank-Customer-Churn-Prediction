import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the machine learning model and encoders
model = joblib.load('XGBoost.pkl')
gender_encode = joblib.load('gender_encode.pkl')
label_encode = joblib.load('label_encode.pkl')

def main():
    st.title('Churn Model Deployment')
    st.header('Fill data down below!')

    # Add user input components for features
    credit_score = st.number_input('Credit Score', min_value=300, max_value=850, value=500)
    geography = st.selectbox('Geography', ['France', 'Germany', 'Spain'])
    gender = st.radio('Gender', ['Male', 'Female'])
    age = st.number_input('Age', min_value=18, max_value=100, value=30)
    tenure = st.number_input('Tenure', min_value=0, max_value=20, value=5)
    balance = st.number_input('Balance', min_value=0.0, value=0.0)
    num_of_products = st.number_input('Number of Products', min_value=1, max_value=4, value=1)
    has_cr_card = st.radio('Has Credit Card', ['Yes', 'No'])
    is_active_member = st.radio('Is Active Member', ['Yes', 'No'])
    estimated_salary = st.number_input('Estimated Salary', min_value=0.0, value=0.0)

   
    # Collect user input data
    input_data = {
        'CreditScore': int(credit_score), 
        'Geography': geography,
        'Gender': gender,
        'Age': int(age),
        'Tenure': int(tenure),
        'Balance': float(balance),
        'NumOfProducts': int(num_of_products),
        'HasCrCard': 1 if has_cr_card == 'Yes' else 0,
        'IsActiveMember': 1 if is_active_member == 'Yes' else 0,
        'EstimatedSalary': float(estimated_salary)
        }

    df=pd.DataFrame([list(input_data.values())], columns=['CreditScore', 'Geography','Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary'])
    
    # Binary Encode for Gender column
    df=df.replace(gender_encode)

    # Label Encode for Geography column
    df['Geography'] = label_encode.fit_transform(df['Geography'])

    
    if st.button('Make Prediction'):
        features=df      
        result = make_prediction(features)
        st.success(f'The prediction is: {result}')
        if result == 1:
          st.write("It indicates that the customer is expected to churn.")
        else:
          st.write("It indicates that the customer is not expected to churn")

def make_prediction(features):
    # Use the loaded model to make predictions
    # Replace this with the actual code for your model
    input_array = np.array(features).reshape(1, -1)
    prediction = model.predict(input_array)
    return prediction[0]

if __name__ == '__main__':
    main()
