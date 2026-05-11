import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
from tensorflow.keras.models import load_model


st.markdown("""
    <style>
    
    .stApp {
        background-color: #B2D8D8 !important;
        color : #355E3B;
    }
    
    div[data-baseweb="input"], 
    div[data-baseweb="select"], 
    .stSlider,
    div[role="radiogroup"] label {
        transition: transform 0.3s ease-in-out, box-shadow 0.3s ease-in-out !important;
    }

    div[data-baseweb="input"]:focus-within, 
    div[data-baseweb="select"]:focus-within {
        transform: scale(1.05);
        box-shadow: 0px 5px 15px rgba(0,0,0,0.1);
        z-index: 10;
    }

    .stSlider:hover {
        transform: scale(1.02);
    }

    div[role="radiogroup"] {
        gap: 20px; /* Adds space so they don't hit each other when scaling */
    }

    div[role="radiogroup"] label:hover {
        transform: scale(1.15); /* Slightly bigger pop for buttons */
        cursor: pointer;
    }
    
    div[role="radiogroup"] label[data-active="true"] {
        transform: scale(1.05);
        font-weight: bold;
    }
    
    div.stButton > button {
        background-color: #00CC99 !important;
        color: black !important;
        border: none !important;
        width: 100% !important;
        font-weight: bold !important;
        border-radius: 8px !important;
    }
    
    </style>
    """, unsafe_allow_html=True)

url = 'http://127.0.0.1:8000/'
st.set_page_config(layout='wide', page_icon='🌮', page_title='Resturent Rating')
@st.cache_resource
def load_network():
    four_scale = joblib.load(r'tools\four_scaler.plk')
    one_scale = joblib.load(r'tools\one_scaler.plk')
    le = joblib.load(r'tools\label_encoder.plk')
    df =   pd.read_csv('dataset/cleaned_df.csv')
    model = load_model('tools/final_model.keras')
    return four_scale,one_scale, le, df, model
    

four_scale, one_scale, le, df, model = load_network()

st.title('Rate the Place of Resturent 🌮', text_alignment='center', width='stretch')

with st.container(border= True, width='stretch') :
    st.subheader('Address:')
    address = st.text_input(' ',placeholder= r'exp. : "942, 21st Main Road, 2nd Stage, Banashankari, Bangalore"')
    
    col1, col2, col3 = st.columns(3)
    
    with col1 :
        st.subheader('Select Resturent Type :')
        resturent_type = st.selectbox('', df['listed_in(type)'].unique())
        resturent_type = le.transform([resturent_type]).tolist()
        
    with col2 :
        st.subheader('Define Food Items : ', text_alignment= 'center')
        food_items = st.text_input('', placeholder= r'exp. : " Momos,Lunch Buffet, Chocolate Nirvana, Paneer Tikka, Dum Biryani"')
        
    with col3 :
        st.subheader('Define Rest Types : ', text_alignment= 'right')
        rest_type = st.text_input('', placeholder= r'exp. : "Casual Dining, Bar" / "Pub"')
        
    
    left, mid, right = st.columns(3)
    
    with left :
        st.subheader('Table Booking:')
        with st.container(border= True, width= 'stretch',height= 100) :
            table_booking = st.radio('', ['Yes', 'No'], key='table', horizontal= True)
            if table_booking == 'Yes' :
                table_booking = 1
            else : 
                table_booking = 0
    with mid :
        st.subheader('Online Order Placing:', text_alignment= 'center')
        with st.container(border= True, width= 'stretch',height= 100) :
            online_order = st.radio('', ['Yes', 'No'], key='online', horizontal= True)
            if online_order == 'Yes' :
                online_order = 1
            else : 
                online_order = 0
    with right :
        st.subheader('Avg cost for two people', text_alignment='right')
        with st.container(border= True, width= 'stretch',height= 100) :
            cost = st.slider('', 200, 2500, step=100)
            
    btn = st.button('Predict', width='stretch')
    if btn :
        st.balloons()
        param = {'text' :  food_items +' ' + rest_type + ' ' + address}
        cost_matrix = np.column_stack([resturent_type,[table_booking], [online_order], [cost]])
        cost_matrix = four_scale.transform(cost_matrix)
        with st.spinner('Thinking ...',show_time= True) :
            response = requests.get( url + 'text_processing/',params= param).json()
            matrix = np.hstack((response,cost_matrix))
            
            result = model.predict(matrix)[0][0]
            scaled = one_scale.inverse_transform(pd.DataFrame([result]).values.reshape(-1,1))
            output = str(round(scaled[0][0].copy(),2))
        st.success(f'This place is { output } /5⭐ according to me. Make a wise decision about it :)')
            
        
        
        