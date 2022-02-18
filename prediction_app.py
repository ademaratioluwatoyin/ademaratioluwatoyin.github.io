# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 23:18:35 2022

@author: Toyin
"""

import numpy as np
import pickle
import streamlit as st
import sklearn
import pandas as pd

df = pd.read_csv('C:/Users/Toyin/Desktop/Model deployment/mobile_price_classification_clean.csv')
# loading the saved model
loaded_model = pickle.load(open('C:/Users/Toyin/Desktop/Model deployment/model.sav', 'rb'))
loaded_scaler = pickle.load(open('C:/Users/Toyin/Desktop/Model deployment/scaler.sav', 'rb'))

# prediction function
def value_prediction(data): 
    
    data = loaded_scaler.fit_transform(np.array(data).reshape(1, -1))
    prediction = loaded_model.predict(data)
    print(prediction)
    if prediction == 0:
        return('value is low')
    elif prediction == 1 or prediction == 2:
        return('value is average')
    else:
        return('value is high')

def main():
    
    st.line_chart(df[['battery_power']])
    
    st.title('Value Prediction App')
    
    battery_power = st.text_input('Battery Power')
    clock_speed = st.text_input('Clock Speed')
    n_cores = st.text_input('Number of Cores')
    
    price_range = ''
    
    if st.button('predict'):
        price_range = value_prediction([battery_power, clock_speed, n_cores])
        
    st.success(price_range)
    
if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    