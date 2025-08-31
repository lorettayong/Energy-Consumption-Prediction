# Project: Energy Consumption Prediction
# Phase 7: Model Deployment using Streamlit Application (app.py)

# 1. Load Additional Libraries
import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import datetime

# 2. Add compatibility shim for XGBoost
# This is a compatbility shim to handle GPU-trained models on a CPU.
# It dynamically removes the gpu_id attribute from the model.
def remove_gpu_attribute(model):
    """
    Removes the gpu_id attribute to allow model loading on a CPU.
    """
    if hasattr(model, 'gpu_id'):
        delattr(model 'gpu_id')
    return model

# 3.1 Load the trained models when the app starts
@st.cache_resource()
def load_model()):
    """
    Loads the trained models from the joblib file and caches them.
    This prevents the models from reloading every time the user interacts with the app.
    """
    try:
        pipeline_models = joblib.load('production_pipeline.joblib')
        # Apply the compatibility shim to the XGBoost model to remove the gpu_id attribute
        if 'xgb_model' in pipeline_models:
            pipeline_models['xgb_model'] = remove_gpu_attribute(pipeline_models['xgb_model'])
        return pipeline_models
    except FileNotFoundError:
        st.error("Error: The 'production_pipeline.joblib' file was not found. Please ensure it exists in the same directory as the app.")
        return None
    except Exception as e:
        st.error("An error has occurred during model loading: {e}.")
        return None

# 3.3 Fetch historical data
@st.cache_data
def fetch_historical_data(end_time, look_back_hours=168):
    """
    Simulates fetching historical data from a database.
    In a real app, this would query a database.
    """
    try:
        # Load the original data files
        pjme_df = pd.read_csv('PJME_hourly.csv', index_col=[0], parse_dates=[0])
        temp_df = pd.read_csv('temp.csv', index_col=[0], parse_dates=[0])
        
        # Merge the two dataframes
        df = pd.merge(pjme_df, temp_df, left_index=True, right_index=True, how='left')

        # Filter for the required historical window
        start_time = end_time - pd.Timedelta(hours=look_back_hours)
        historical_df = df.loc[start_time:end_time]

        return historical_df
    except FileNotFoundError:
        st.error("Error: Historical data files (PJME_hourly.csv, temp.csv) not found. Please ensure they are in the same directory.")
        return None
    except Exception as e:
        st.error(f"An error occurred while fetching historical data: {e}")
        return None

# 3.3 Replicate the exact feature engineering steps
def make_prediction(models, raw_data:
    """
    Takes a new raw data, fetches historical data, engineers all features, and returns a single energy consumption prediction.
    """
    if models is None:
        return None
    
    try:
        # Convert the single input into a DataFrame and set index
        df = pd.DataFrame([raw_data])
        df.set_index(pd.to_datetime(df['Datetime']), inplace=True)

        # Fetch historical data ending at the prediction time
        historical_df = fetch_historical_data(df.index[0])

        # Combine the historical data with the new data point for feature engineering
        full_df = pd.concat([historical_df, df])
        full_df = full_df[~full_df.index.duplicated(keep='first')] # Remove any duplicate indices
        full_df.sort_index(inplace=True)

        # Add feature engineering
        full_df['hour_of_day'] = full_df.index.hour
        full_df['day_of_week'] = full_df.index.dayofweek
        full_df['day_of_year'] = full_df.index.dayofyear
        full_df['week_of_year'] = full_df.index.isocalendar().week.astype(int)
        full_df['month'] = full_df.index.month
        full_df['quarter'] = full_df.index.quarter
        full_df['year'] = full_df.index.year
        full_df['is_weekend'] = (full_df.index.dayofweek >= 5).astype(int)
        full_df['day_of_month'] = full_df.index.day

        # Add cyclical features
        full_df['hour_sin'] = np.sin(2 * np.pi * full_df['hour_of_day'] / 24)
        full_df['hour_cos'] = np.cos(2 * np.pi * full_df['hour_of_day'] / 24)
        full_df['dayofyear_sin'] = np.sin(2 * np.pi * full_df['day_of_year'] / 365)
        full_df['dayofyear_cos'] = np.cos(2* np.pi * full_df['day_of_year'] / 365)

        # Recreate lagged features
        full_df['lag_1_hour'] = full_df['PJME_MW'].shift(1)
        full_df['lag_24_hour'] = full_df['PJME_MW'].shift(24)
        full_df['lag_168_hour'] = full_df['PJME_MW'].shift(168)

        # Recreate rolling window statistics
        full_df['PJME_MW_rolling_24_hr_mean'] = full_df['PJME_MW'].rolling(window='24h', closed='left').mean().bfill()
        full_df['PJME_MW_rolling_168_hr_mean'] = full_df['PJME_MW'].rolling(window='168h', closed='left').mean().bfill()
        full_df['PJME_MW_rolling_24_hr_std'] = full_df['PJME_MW'].rolling(window='24h', closed='left').std().bfill()
        full_df['PJME_MW_rolling_168_hr_std'] = full_df['PJME_MW'].rolling(window='168h', closed='left').std().bfill()

        # Recreate temperature-based features
        full_df['lag_24_temp'] = full_df['temperature'].shift(24)
        full_df['rolling_72_temp_avg'] = full_df['temperature'].rolling(window=72).mean()
        full_df['temp_squared'] = np.power(full_df['temperature'], 2)

        # Recreate interaction and other features
        full_df['hour_of_day_x_is_weekend'] = full_df['hour_of_day'] * (full_df['is_weekend'] + 1)
        full_df['temperature_x_is_holiday'] = full_df['temperature'] * (full_df['is_holiday'] + 1)
        full_df['temperature_x_hour_of_day'] = full_df['temperature'] * (full_df['hour_of_day'] + 1)
        full_df['cdd_x_is_weekend'] = full_df['cdd'] * full_df['is_weekend']

        # A single data point to predict is the last row after all engineering steps
        X = full_df.iloc[-1].to_frame().T

        # The final list of features
        FEATURES = [
            'hour_of_day', 'day_of_week', 'day_of_year', 'week_of_year', 'month', 'quarter', 'year', 'is_weekend',
            'day_of_month', 'hour_sin', 'hour_cos', 'dayofyear_sin', 'dayofyear_cos', 'lag_1_hour', 'lag_24_hour',
            'lag_168_hour', 'is_holiday', 'temperature', 'lag_24_temp', 'rolling_72_temp_avg', 'temp_squared',
            'hdd', 'cdd', 'PJME_MW_rolling_24_hr_mean', 'PJME_MW_rolling_168_hr_mean',
            'PJME_MW_rolling_24_hr_std', 'PJME_MW_rolling_168_hr_std', 'hour_of_day_x_is_weekend',
            'temperature_x_is_holiday', 'temperature_x_hour_of_day', 'cdd_x_is_weekend'
        ]

        X = X[FEATURES]
        X.columns = X.columns.astype(str) # ensure column names are in strings

        # Generate predictions from the base models
        xgb_preds = trained_models['xgb_model'].predict(X)
        lgbm_preds = trained_models['lgbm_model'].predict(X)

        # Use the meta-model for the final prediction
        stacked_features = pd.DataFrame({
            'xgb_preds': xgb_preds,
            'lgbm_preds': lgbm_preds
        })

        final_prediction = trained_models['meta_model'].predict(stacked_features)

        return final_prediction[0]

    except Exception as e:
        st.error(f"An error has occurred during prediction: {e}")
        return None

# 2.5 Create Streamlit UI
st.title("Energy Consumption Forecasting")
st.markdown("Enter the details below to get a real-time energy consumption prediction.")

# Load models
models = load_model()

if models:
    st.sidebar.header("Input Parameters")
    
    # Collect user inputs
    prediction_date = st.sidebar.date_input(
        "Date", datetime.date.today()
    )
    prediction_time = st.sidebar.time_input(
        "Time", datetime.time(12, 0)
    )

    current_temp = st.sidebar.slider(
        "Current Temperature (°C)", -10.0, 40.0, 18.0
    )

    # Combine date and time
    datetime_combined = datetime.datetime.combine(prediction_date, prediction_time)

    user_input = {
        'Datetime': datetime_combined,
        'current_temp': current_temp,
        'PJME_MW': 0, # Placeholder value for the prediction target
        'current_temp': current_temp # Ensure current_temp is passed to the function
    }
    
    if st.sidebar.button("Predict"):
        with st.spinner('Making prediction...'):
            prediction = make_prediction(models, user_input)
            if prediction is not None:
                st.subheader("Prediction Results")
                st.success(f"The predicted energy consumption is: **{prediction:,.2f} MW**")
                
                # Simple explanation for the user
                st.markdown("""
                <div style="background-color:#f0f2f6;padding:10px;border-radius:10px;">
                    <p>The prediction is based on a powerful stacking ensemble model. It combines the strengths of multiple machine learning models to provide a more accurate forecast. The model considered factors like time of day, day of the week, seasonality, temperature, and historical consumption data to make this prediction.</p>
                </div>
                """, unsafe_allow_html=True)