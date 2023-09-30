
import numpy as np
import pandas as pd 
import streamlit as st
import joblib
import sklearn


states = ['Puerto Rico', 'Virgin Islands', 'Massachusetts', 'Connecticut',
       'New Hampshire', 'Vermont', 'New Jersey', 'New York',
       'Rhode Island', 'Maine', 'Pennsylvania', 'Delaware']

price_category = ['Affordable', 'Moderate', 'Luxury']

new = ['yes', 'no']


model = joblib.load('model.h5')
inputs = joblib.load('inputs.h5')
inputs = inputs.tolist()

def prediction(bedrooms,bathrooms,land_area_in_acres,state,housearea_sqft,price_per_sqft,Price_Category, Not_Brand_New):
    test_df = pd.DataFrame(columns=inputs)
    test_df.at[0, 'bedrooms'] = bedrooms
    test_df.at[0, 'bathrooms'] = bathrooms
    test_df.at[0, 'land_area_in_acres'] = land_area_in_acres
    test_df.at[0, 'state'] = state
    test_df.at[0, 'housearea_sqft'] = housearea_sqft
    test_df.at[0, 'price_per_sqft'] = price_per_sqft
    test_df.at[0, 'Price Category'] = Price_Category
    test_df.at[0, 'Not Brand New'] = np.where(Not_Brand_New == 'yes', 1, 0)
    result = model.predict(test_df)
    result = int(np.exp(result))
    return result


def main():
    st.markdown("<h1 style='text-align: center; color: white;'>US House_Prices Predictor</h1>", unsafe_allow_html=True)
    bedrooms = st.slider('bedrooms',1, 15)
    bathrooms = st.slider('bathrooms',1, 15)
    land_area_in_acres = st.slider('land_area_in_acres',0.1, 18.0)
    state = st.selectbox('state', states)
    housearea_sqft = st.slider('housearea_sqft',100, 4000)
    price_per_sqft = st.slider('price_per_sqft',13, 6150)
    Price_Category = st.selectbox('Price_Category', price_category)
    Not_Brand_New = st.selectbox('Not_Brand_New', new)

    if st.button('Prediction'):
        results = prediction(bedrooms,bathrooms,land_area_in_acres,state,housearea_sqft,price_per_sqft,Price_Category, Not_Brand_New)
        st.text(f'Predicted House Price: {results} USD')

if __name__=='__main__':
    main() 
