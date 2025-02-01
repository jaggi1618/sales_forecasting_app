import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.metrics import mean_squared_error

# Load the saved model
model = joblib.load('xgboost_sales_model.pkl')
data = pd.read_csv("extended_data_20000_with_lags.csv") 

# Streamlit app title
st.title("Sales Forecasting App")
st.markdown("Dataset : ")
st.dataframe(data)

# User input for future days to forecast
num_days_to_forecast = st.number_input("Enter number of days to forecast:", min_value=1, max_value=365, value=30)

# Function to forecast future values
def forecast_future_values(model, data, num_days_to_forecast):

    last_date = pd.to_datetime(data['date'].iloc[-1])
    future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, num_days_to_forecast + 1)]
    future_data = pd.DataFrame({'date': future_dates})
    
    # Generate future features
    future_data['day'] = future_data['date'].dt.day
    future_data['month'] = future_data['date'].dt.month
    future_data['year'] = future_data['date'].dt.year
    future_data['weekday'] = future_data['date'].dt.weekday
    future_data['week_of_year'] = future_data['date'].dt.isocalendar().week
    future_data['quarter'] = future_data['date'].dt.quarter
    
    # Using the most recent values from the original data for rolling statistics
    future_data['money_rolling_mean_7'] = data['money'].rolling(window=7).mean().iloc[-1]
    future_data['money_rolling_std_7'] = data['money'].rolling(window=7).std().iloc[-1]
    future_data['money_rolling_mean_30'] = data['money'].rolling(window=30).mean().iloc[-1]
    future_data['money_rolling_std_30'] = data['money'].rolling(window=30).std().iloc[-1]
    
    # Add Fourier series terms for seasonality
    t = np.arange(len(data) + num_days_to_forecast)
    for i in range(1, 4):
        future_data[f'sin_{i}'] = np.sin(2 * np.pi * i * t[-num_days_to_forecast:] / 365.25)
        future_data[f'cos_{i}'] = np.cos(2 * np.pi * i * t[-num_days_to_forecast:] / 365.25)
    
    # Select and standardize features
    features = ['day', 'month', 'year', 'weekday', 'week_of_year', 'quarter',
                'money_rolling_mean_7', 'money_rolling_std_7',
                'money_rolling_mean_30', 'money_rolling_std_30',
                'sin_1', 'cos_1', 'sin_2', 'cos_2', 'sin_3', 'cos_3']
    scaler = StandardScaler()
    X_future = scaler.fit_transform(future_data[features])
    
    # Predict using the trained model
    future_data['predicted_money'] = model.predict(X_future)
    return future_data[['date', 'predicted_money']]

if st.button("Forecast"):
    forecasted_data = forecast_future_values(model, data, num_days_to_forecast)
    forecasted_data['date'] = pd.to_datetime(forecasted_data['date'])  # Ensure 'date' is in datetime format
    st.write("Forecasted Values:")
    st.dataframe(forecasted_data)

    # Plotting the forecasted data
    st.subheader("Forecast Plot")

    # Line plot of the forecasted values
    plt.figure(figsize=(10, 6),dpi=200)
    plt.plot(pd.to_datetime(data['date']), data['money'], label='Historical Data', color='blue')
    plt.plot(forecasted_data['date'], forecasted_data['predicted_money'], label='Forecast', color='orange')
    plt.xlabel("Date")
    plt.ylabel("Money")
    plt.title("Historical and Forecasted Money Values")
    plt.legend()
    plt.grid(visible=True, linestyle='--', alpha=0.5)
    st.pyplot(plt)

    # Additional plots if needed
    st.subheader("Additional Visualizations")

    # Bar chart of the forecasted values
    st.bar_chart(forecasted_data.set_index('date')['predicted_money'])
