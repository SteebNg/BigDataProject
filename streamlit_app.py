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

# --- Main Streamlit App Structure ---
def main():
    st.set_page_config(
        page_title="Disease Trends in Malaysia",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # --- Header and Introduction ---
    st.image("INTI_Logo.png")

    st.title("🇲🇾 Healthcare Data Insights: Disease Trends in Malaysia")
    st.markdown("""
    This interactive dashboard, developed for the **5011CEM Big Data Programming Project**,
    provides insights into disease patterns and prevalence in Malaysia.
    Leveraging Big Data principles, it categorizes diseases, analyzes their trends over time (2017-2021),
    and highlights geographical areas with higher disease prevalence.
    """)

    st.markdown("---")

    # --- Sidebar for Navigation and Filters ---
    st.sidebar.header("Navigation & Filters")
    analysis_options = [
        "Dashboard Overview",
        "Disease Category Trends",
        "Geographical Analysis",
        "Predictive Analysis",
        "Data Explorer",
        "About This Project"
    ]
    selected_analysis = st.sidebar.radio("Go to:", analysis_options)

    # --- Data Loading and Preprocessing ---
    # Perform these steps once and cache the results
    with st.spinner('Loading and pre-processing data...'):
        df_raw = load_data(DATASET_PATH)
        processed_df = preprocess_data(df_raw)
        yearly_category_trends = get_yearly_category_trends(processed_df)
        state_category_trends = get_state_category_trends(processed_df)
        overall_category_summary = get_overall_category_summary(processed_df)
    st.sidebar.success("Data Ready!")

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

