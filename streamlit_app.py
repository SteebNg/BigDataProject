import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os # To handle directory creation
import numpy as np # Ensure numpy is imported
from sklearn.linear_model import LinearRegression

# --- Configuration and Constants ---
DATASET_PATH = 'std_state.csv'
OUTPUT_DIR = 'temp_streamlit_plots/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Refined Disease Categories (YOU MUST VERIFY AND JUSTIFY THESE IN YOUR REPORT!)
# This is a critical research component for your assignment.
DISEASE_CATEGORIES = {
    'chancroid': 'Chancroid',
    'gonorrhea': 'Gonorrhea',
    'hiv': 'Hiv',
    'syphillis': 'Syphillis',
    'aids': 'Aids'
}

# --- NEW: Machine Learning Functions (Linear Regression Forecasting) ---
@st.cache_resource # Use st.cache_resource for models
def train_linear_regression_model(df_filtered):
    """
    Trains a Linear Regression model for forecasting.
    df_filtered must contain 'year' and 'total_cases'.
    """
    if df_filtered.empty or len(df_filtered) < 2:
        return None, "Not enough data points for linear regression (at least 2 required)."

    X = df_filtered[['year']]
    y = df_filtered['total_cases']

    model = LinearRegression()
    model.fit(X, y)
    return model, None

def make_linear_regression_forecast(model, last_year, years_to_forecast):
    """
    Generates future dates and makes predictions using a trained Linear Regression model.
    """
    future_years = np.array(range(last_year + 1, last_year + 1 + years_to_forecast)).reshape(-1, 1)
    forecasted_cases = model.predict(future_years)

    # Create a DataFrame for display
    forecast_df = pd.DataFrame({
        'Year': future_years.flatten(),
        'Predicted Total Cases': forecasted_cases
    })
    return forecast_df

if __name__ == "__main__":
    sns.set_style("whitegrid")
    # Set a larger default font size for plots
    plt.rcParams.update({'font.size': 10})
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.dpi'] = 100 # Adjust for better resolution if needed

    main()

#BREAAAAAAAAAAKKAKAKAKAKAAKKAAKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKK
