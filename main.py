import streamlit as st
import pickle as pkl
import numpy as np

pipe = pkl.load(open('pipe.pkl', 'rb'))
data = pkl.load(open('df.pkl', 'rb'))

st.title('Laptop Price Prediction App')

# Select brand
company = st.selectbox('Brand', data['Company'].unique())

# Select type of laptop
type = st.selectbox('Type of Laptop', data['TypeName'].unique())

# Select RAM
ram = st.selectbox('RAM (GB)', [2, 4, 6, 8, 12, 16])

# Laptop weight
weight = st.number_input('Weight of the laptop (kg)')

# Touchscreen
touchscreen = st.selectbox('Touchscreen', ['Yes', 'No'])

# IPS Display
IPS = st.selectbox('IPS', ['Yes', 'No'])

# Screen size
screensize = st.number_input('Screen Size (inches)')

# Screen resolution
resolution = st.selectbox('Screen Resolution', ['1920x1080', '1366x768', '1600x900', '3040x2160',
                                                '3200x1800', '2880x1800', '2560x1600',
                                                '2560x2440', '2304x1440'])

# CPU brand
cpu = st.selectbox('CPU Brand', data['Cpu brand'].unique())

# HDD size
HDD = st.selectbox('HDD (GB)', [0, 128, 256, 512, 1024, 2048])

# SSD size
SSD = st.selectbox('SSD (GB)', [0, 8, 128, 256, 512, 1024])

# GPU brand
gpu = st.selectbox('GPU Brand', data['GPU brand'].unique())

# Operating System
os = st.selectbox('OS', data['OS'].unique())

if st.button('Predict Value'):
    # Convert categorical inputs
    IPS = 1 if IPS == 'Yes' else 0
    touchscreen = 1 if touchscreen == 'Yes' else 0

    # Calculate PPI
    x_res = int(resolution.split('x')[0])
    y_res = int(resolution.split('x')[1])
    ppi = ((x_res**2 + y_res**2) ** 0.5) / screensize

    # Prepare the input array for the model
    query = np.array([company, type, ram, weight, touchscreen, IPS, ppi, cpu, HDD, SSD, gpu, os])

    # Ensure the query is the correct shape for the model
    query = query.reshape(1, -1)

    # Predict and display the result
    predicted_price = np.exp(pipe.predict(query)[0])
    st.title(f'Predicted Price For similar configration:{predicted_price:.2f}')









