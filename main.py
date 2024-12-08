import pickle
import numpy
import sklearn
import streamlit as st
from function import load_models,predict_salary, preprocess_input,load_cal
import main_test

path = './Salary Data.csv'

# edu_levels, job_title = load_cal(path)

# lr_model, nn_model, scaler, encoder = load_models()

# st.title("Salary prediction APP")
# st.write("This app is to predict the salary")

# age = st.number_input("Age",min_value=0, max_value=100, value=25)
# gender = st.selectbox("Gender",["Male","Female"])
# education_level = st.text_input("Education level")
# job_title = st.text_input("Job title")
# year_exp = st.number_input("Year",min_value=0, max_value=100, value=25)
# print("0")
# if st.button("Predict Salary"):
#     try:
#         # Preprocess input
#         input_data = preprocess_input(age, gender, education_level, job_title, year_exp, encoder, scaler)
        
#         # Predict salaries using both models
#         salary_lr = predict_salary(lr_model,input_data)
#         salary_nn = predict_salary(nn_model, input_data)

#         # Display results
#         st.success(f"Salary predicted by Linear Regression Model: ${salary_lr:,.2f}")
#         st.success(f"Salary predicted by Neural Network Model: ${salary_nn:,.2f}")
#     except Exception as e:
#         st.error(f"An error occurred: {e}")
#main()
# if __name__ == "__main__":
#     main()