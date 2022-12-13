import streamlit as st
import numpy as np
import pandas as pd
from Predict import *
import GatherData as gd
st.set_option('deprecation.showPyplotGlobalUse', False)

mission = ['TESS', 'Kepler']
st.title('Exoplanet Hunter')
st.header('A Tool to analyze/validate Exoplanet Data')

st.sidebar.title('Choose a starting point')
selected_mission = st.sidebar.selectbox('',mission)
DataObject = gd.GatherData(mission=selected_mission)
input_ids = DataObject.IdData['WrapperId']

selected_id = st.sidebar.selectbox(f'Available IDs for {selected_mission}', 
input_ids
)


#Predict(mission = 'Kepler', ID=10811496)
if st.sidebar.button('Run ExoMiner Raw Model'):
    try:
        prediction = Predict(selected_mission, selected_id)
        formatted_prediction = f"<p class=\"colored-font\"> Model output: <span style=\"color: black\"> </span><span style=\"color: green\"> {prediction} </span></p>"

        st.markdown(formatted_prediction, unsafe_allow_html=True)
        #st.write(Predict(mission = 'Kepler', ID=10811496))

        import matplotlib.pyplot as plt
        import numpy as np
        fig, ax = plt.subplots()
        st.pyplot(gd.KeplerTest(selected_id))
    except:
        st.warning('Unable to fetch/process data for the selected ID, please select another ID.', icon="⚠️")


