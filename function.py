import pickle
import numpy as np
import sklearn
import pandas as pd

def load_models():
    with open("lr_model.pkl",'rb') as lr_file:
        lr_model = pickle.load(lr_file)
    with open("nn_model.pkl",'rb') as nn_file:
        nn_model = pickle.load(nn_file)
    with open("scaler.pkl",'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    with open("encoder.pkl",'rb') as encoder_file:
        encoder = pickle.load(encoder_file)
    return lr_model, nn_model, scaler, encoder
# 
# def preprocess_input(age,gender,education_level,job_title,year_exp,encoder, scaler):
#     cal_data = [[gender,education_level,job_title]]
#     encoded_data = encoder.transform(cal_data)

#     num_data = np.array([[age,year_exp]])
#     full_data = np.hstack((num_data,encoded_data))

#     scaled_data = scaler.transform(full_data)

#     return scaled_data
def preprocess_input(age, gender, education_level, job_title, year_exp, encoder, scaler):
    # Transform categorical data using the encoder
    categorical_data = [[gender, education_level, job_title]]  # Ensure it's a 2D array
    encoded_data = encoder.transform(categorical_data)  # Apply one-hot encoding
    
    # Prepare numerical data
    numerical_data = np.array([[age, year_exp]])  # Ensure it's a 2D array
    
    # Combine numerical and encoded categorical data
    full_data = np.hstack((numerical_data, encoded_data))
    
    # Scale the combined data
    scaled_data = scaler.transform(full_data)  # Ensure scaling
    return scaled_data


def load_cal(path):
    data = pd.read_csv(path)
    edu_levels = data['Education Level'].unique().tolist()
    job_title = data['Job Title'].unique().tolist()
    return edu_levels, job_title

def predict_salary(model,input_data):
    return model.predict(input_data)[0]