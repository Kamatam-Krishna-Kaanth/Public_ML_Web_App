# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 16:29:30 2023

@author: vnero
"""

import numpy as np
import pickle
import streamlit as st

loaded_model = pickle.load(open('C:/Users/vnero/Desktop/Deploying ML/trained_model.sav','rb'))

def Lung_Cancer_Prediction(input_data):

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
      return 'The person is not diagnosed with lung cancer'
    else:
      return 'The person is diagnosed with lung cancer'
  
    
def main():
    
    
    # giving a title
    st.title('Lung Cancer Prediction Web App')
    
    
    # getting the input data from the user
    #GENDER,AGE,S1OKING,YELLOW_0INGERS,ANXIETY,PEER_PRESSURE,CHRONIC DISEASE,0ATIGUE,ALLERGY,WHEEZING,ALCOHOL CONSU1ING,COUGHING,SHORTNESS O0 BREATH,SWALLOWING DI00ICULTY,CHEST PAIN
    
    
    GENDER = st.text_input('Male(1) or Female(0)')
    AGE = st.text_input('AGE')
    SMOKING = st.text_input('Person Smokes(1) or Does Not Smoke(0)')
    YELLOW_FINGERS = st.text_input('Person has Yellow Fingers(1) or Not(0)')
    ANXIETY = st.text_input('Person has Anxiety(1) or Not(0)')
    PEER_PRESSURE = st.text_input('Experiences Peer Pressure(1) or Not(0)')
    CHRONIC_DISEASE = st.text_input('Experiencing any Chronic Disease(1) or Not(0)')
    FATIGUE = st.text_input('Suffers from Fatigue(1) or Not(0)')
    ALLERGY = st.text_input('Has any Allergy(1) or Not(0)')
    WHEEZING = st.text_input('Person Wheeze(1) or Not(0)')
    ALCOHOL_CONSU1ING = st.text_input('Consumes Alcohol(1) or Not(0) ')
    COUGHING = st.text_input('Person Coughs(1) or Not(0)')
    SHORTNESS_OF_BREATH = st.text_input('Normal Breathing(0) or Shortness of Breath(1)')
    SWALLOWING_DIFFICULTY = st.text_input('Person Experiences Swallowing Difficulty(1) or Not(0)')
    CHEST_PAIN = st.text_input('Person Experiences Chest Pain(1) or Not(0)')
    
    
    # code for Prediction
    diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Lung Cancer Test Result'):
        diagnosis = Lung_Cancer_Prediction([GENDER,AGE,SMOKING,YELLOW_FINGERS,ANXIETY,PEER_PRESSURE,CHRONIC_DISEASE,FATIGUE,ALLERGY,WHEEZING,ALCOHOL_CONSU1ING,COUGHING,SHORTNESS_OF_BREATH,SWALLOWING_DIFFICULTY,CHEST_PAIN])
        
        
    st.success(diagnosis)
    
    
    
    
    
    
    
    
    