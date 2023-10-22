# Predicting US Real Estate Prices with Machine Learning
In this project, I delve into the fascinating world of real estate and leverage the power of machine learning to predict property prices across the United States.
This repository serves as a central hub for all the code, data, and documentation related to our undertaking.
There are 2 datasets each with 1 CSV file

Acknowledgements
Data was collected from -
https://www.realtor.com/ - A real estate listing website operated by the News Corp subsidiary Move, Inc. and based in Santa Clara, California. It is the second most visited real estate listing website in the United States as of 2021, with over 100 million monthly active users.

# Uncleaned dataset:
real_estate_USA.csv (100k+ entries)
status (Housing status - a. ready for sale or b. ready to build)
bed (# of beds)
bath (# of bathrooms)
acre_lot (Property / Land size in acres)
city (city name)
state (state name)
zip_code (postal code of the area)
house_size (house area/size/living space in square feet)
prev_sold_date (Previously sold date)
price (Housing price, it is either the current listing price or recently sold price if the house is sold recently)

# Cleaned Dataset:
clean_realestate_USA.csv
- for simple use of Pipeline function for cross validation
- for simple and fast model implementations

# Machine Learning Techniques:
To achieve the goal, I tested a variety of machine learning techniques, including but not limited to:
- Regression Models: Linear regression, decision tree regression, random forest regression and more to model the relationship between features and property prices.
- Data Preprocessing: Data cleaning, feature engineering, and normalization to ensure high-quality input for our models.
- Model Evaluation: Rigorous evaluation using metrics like mean squared error (MSE), and R-squared to assess model performance.


# Disclaimer:
The data and information in the data set provided here are intended to use for educational purposes only. I do not own any data, and all rights are reserved to the respective owners.
