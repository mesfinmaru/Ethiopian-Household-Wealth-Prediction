"""
Ethiopian Household Wealth Prediction Dashboard
================================================
Streamlit dashboard for interactive wealth prediction.

To run:
    cd app
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.path.abspath('..'))
from src.inference import WealthPredictorAPI

st.set_page_config(page_title="Ethiopian Wealth Predictor", page_icon="🇪🇹", layout="wide")

@st.cache_resource
def load_api():
    return WealthPredictorAPI(model_path='../models/')

api = load_api()

# Sidebar
with st.sidebar:
    st.title("🇪🇹 Wealth Predictor")
    page = st.radio("Menu", ["🏠 Home", "🔮 Prediction", "🔬 What-If"])

# Home
if page == "🏠 Home":
    st.title("Ethiopian Household Wealth Predictor")
    st.markdown("""
    Predict household consumption expenditure using machine learning
    trained on 5 waves of Ethiopian Socioeconomic Survey data (2011-2022).
    
    **Features:** Single prediction, What-if scenarios, Regional analysis
    **Model:** XGBoost / LightGBM ensemble
    
    ⚠️ COVID-19 effects and Tigray conflict impacts are incorporated from Wave 5.
    """)

# Prediction
elif page == "🔮 Prediction":
    st.title("Household Wealth Prediction")
    c1, c2, c3 = st.columns(3)
    with c1:
        hh_size = st.slider("Household Size", 1, 15, 4)
        head_age = st.slider("Head Age", 18, 90, 35)
        head_gender = st.selectbox("Head Gender", ["Male", "Female"])
    with c2:
        edu = st.slider("Education Years", 0, 20, 6)
        lit = st.slider("Literacy Rate", 0.0, 1.0, 0.5)
        rooms = st.slider("Rooms", 1, 8, 2)
    with c3:
        elec = st.checkbox("Electricity", True)
        water = st.checkbox("Water", True)
        covid = st.checkbox("Post-COVID", True)
    assets = st.multiselect("Assets", 
        ['Radio','TV','Mobile Phone','Refrigerator','Bicycle','Motorcycle','Car','Computer','Stove'],
        ['Mobile Phone','Radio'])
    region = st.selectbox("Region", 
        ['Addis Ababa','Oromia','Amhara','Tigray','SNNP','Somali','Afar','Harari','Dire Dawa'])

    if st.button("Predict", type="primary", use_container_width=True):
        r = api.predict_single(
            hh_size=hh_size, head_age=head_age, head_gender=1 if head_gender == "Male" else 0,
            education_years=edu, literacy_rate=lit, rooms=rooms,
            has_electricity=int(elec), has_water=int(water),
            asset_owned=[a.lower() for a in assets], region=region, post_covid=int(covid)
        )
        if 'error' not in r:
            c1, c2, c3 = st.columns(3)
            c1.metric("Annual (ETB)", f"{r['consumption_etb']:,.0f}")
            c2.metric("Per Capita", f"{r['per_capita_etb']:,.0f}")
            c3.metric("Category", r['wealth_category'])

# What-If
elif page == "🔬 What-If":
    st.title("What-If Analysis")
    vary = st.selectbox("Vary", ['education_years', 'hh_size', 'head_age', 'rooms', 'asset_count'])
    base = {'hh_size': 4, 'head_age': 35, 'head_gender': 1, 'education_years': 6, 'literacy_rate': 0.5,
            'rooms': 2, 'has_electricity': 1, 'has_water': 1,
            'asset_owned': ['mobile', 'radio', 'bed'], 'region': 'Oromia', 'post_covid': 1}
    
    if vary == 'education_years':
        vals = range(0, 21, 4)
    elif vary == 'hh_size':
        vals = range(1, 15, 2)
    elif vary == 'head_age':
        vals = range(20, 81, 15)
    elif vary == 'rooms':
        vals = range(1, 9)
    else:
        vals = range(0, 15, 2)

    if st.button("Run Analysis", type="primary"):
        df = api.what_if(base, vary, vals)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(df[vary], df['per_capita_etb'], 'o-', linewidth=2, markersize=8, color='steelblue')
        ax.set_xlabel(vary); ax.set_ylabel('Per Capita (ETB)')
        ax.set_title(f'Impact of {vary} on Wealth'); ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        st.dataframe(df[['per_capita_etb', 'wealth_category', vary]].round(0))