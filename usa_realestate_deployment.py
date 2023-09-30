
import numpy as np
import pandas as pd 
import streamlit as st
import joblib
import math
from sklearn.tree import DecisionTreeRegressor

states = ['Puerto Rico', 'Virgin Islands', 'Massachusetts', 'Connecticut',
       'New Hampshire', 'Vermont', 'New Jersey', 'New York',
       'Rhode Island', 'Maine', 'Pennsylvania', 'Delaware']

price_category = ['Affordable', 'Moderate', 'Luxury']

new = ['yes', 'no']


model = joblib.load('model.pkl')
inputs = joblib.load('inputs.pkl')
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
    result = int(math.exp(result))
    return result


def main():

    bedrooms = st.sidebar.slider('bedrooms',1, 15)
    bathrooms = st.sidebar.slider('bathrooms',1, 15)
    land_area_in_acres = st.sidebar.slider('land_area_in_acres',0.1, 18.0)
    state = st.sidebar.selectbox('state', states)
    housearea_sqft = st.sidebar.slider('housearea_sqft',100, 4000)
    price_per_sqft = st.sidebar.slider('price_per_sqft',13, 6150)
    Price_Category = st.sidebar.selectbox('Price_Category', price_category)
    Not_Brand_New = st.sidebar.selectbox('Not_Brand_New', new)

    if st.button('Prediction'):
        results = prediction(bedrooms,bathrooms,land_area_in_acres,state,housearea_sqft,price_per_sqft,Price_Category, Not_Brand_New)
        st.text(f'Predicted House Price: {results}')

if __name__=='__main__':
    main() 
