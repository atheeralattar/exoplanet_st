import streamlit as st
import numpy as np
import pandas as pd
from Predict import *
import GatherData as gd

mission = ['K2', 'TESS', 'Kepler']
st.title('Exoplanet Hunter')
st.header('A Tool to analyze/validates Exoplanets Data')

st.sidebar.title('Choose a starting point')
selected_mission = st.sidebar.selectbox('',mission)
DataObject = gd.GatherData(mission=selected_mission)
input_ids = DataObject.IdData['WrapperId']

selected_id = st.sidebar.selectbox(f'Available IDs for {selected_mission}', 
input_ids
)


#Predict(mission = 'Kepler', ID=10811496)
if st.sidebar.button('Run ExoMiner Raw Model'):
    #Predict.Predict(selected_mission, selected_id)
    Predict(mission = 'Kepler', ID=10811496)



