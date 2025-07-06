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


# Refined Disease Categories
DISEASE_CATEGORIES = {
    'chancroid': 'Chancroid',
    'gonorrhea': 'Gonorrhea',
    'hiv': 'Hiv',
    'syphillis': 'Syphillis',
    'aids': 'Aids'
}

# --- 1. Data Loading and Initial Pre-processing ---
@st.cache_data # Cache this function to run only once for efficiency
def load_data(filepath):
    """
    Loads the dataset from a CSV file.
    """
    try:
        df = pd.read_csv(filepath)
        return df
    except FileNotFoundError:
        st.error(f"Error: Dataset not found at {filepath}. Please ensure '{filepath}' is in the same directory as the app.")
        st.stop() # Stop execution if data loading fails
    except Exception as e:
        st.error(f"An error occurred during data loading: {e}. Please check your dataset format.")
        st.stop() # Stop execution if data loading fails

@st.cache_data # Cache this function as well for efficiency
def preprocess_data(df):
    """
    Performs initial data cleaning and necessary transformations.
    - Converts 'date' column to datetime objects.
    - Adds a 'year' column.
    - Adds 'disease_category' based on predefined mapping.
    - Basic data quality checks.
    """
    df_processed = df.copy() # Work on a copy to avoid SettingWithCopyWarning

    # Convert 'date' to datetime and extract 'year'
    df_processed['date'] = pd.to_datetime(df_processed['date'])
    df_processed['year'] = df_processed['date'].dt.year

    # Add disease categories
    df_processed['disease_category'] = df_processed['disease'].map(DISEASE_CATEGORIES)

    # Handle diseases not found in the mapping
    unmapped_diseases = df_processed[df_processed['disease_category'].isnull()]['disease'].unique()
    if len(unmapped_diseases) > 0:
        st.sidebar.warning(f"Unmapped Diseases: {', '.join(unmapped_diseases)}. Assigned to 'Other/Unspecified'. Please refine mapping.")
        df_processed['disease_category'].fillna('Other/Unspecified', inplace=True)
    
    # Basic Data Quality Checks: Check for non-negative cases and incidence
    if (df_processed['cases'] < 0).any():
        st.sidebar.warning("Negative 'cases' values found and set to 0.")
        df_processed.loc[df_processed['cases'] < 0, 'cases'] = 0
    if (df_processed['incidence'] < 0).any():
        st.sidebar.warning("Negative 'incidence' values found and set to 0.")
        df_processed.loc[df_processed['incidence'] < 0, 'incidence'] = 0

    return df_processed

# --- 2. Data Analysis Functions ---
@st.cache_data
def get_yearly_category_trends(df):
    return df.groupby(['year', 'disease_category']).agg(
        total_cases=('cases', 'sum'),
        average_incidence=('incidence', 'mean')
    ).reset_index()

@st.cache_data
def get_state_category_trends(df):
    return df.groupby(['state', 'disease_category']).agg(
        total_cases=('cases', 'sum'),
        average_incidence=('incidence', 'mean')
    ).reset_index()

@st.cache_data
def get_overall_category_summary(df):
    return df.groupby('disease_category').agg(
        total_cases=('cases', 'sum'),
        avg_incidence=('incidence', 'mean')
    ).sort_values(by='total_cases', ascending=False).reset_index()

# --- 3. Visualization Functions ---
# To visualize disease trends over time
def plot_cases_over_time(df, title):
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=df, x='year', y='total_cases', hue='disease_category', marker='o', ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Year')
    ax.set_ylabel('Total Cases')
    ax.set_xticks(df['year'].unique())
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(title='Disease Category', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    return fig

# To visualize the average incidence rates of diseases over time
def plot_incidence_over_time(df, title):
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=df, x='year', y='average_incidence', hue='disease_category', marker='o', ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Year')
    ax.set_ylabel('Average Incidence')
    ax.set_xticks(df['year'].unique())
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(title='Disease Category', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    return fig

# To visualize the total number of cases for each disease category across all years and states
def plot_overall_cases_by_category(df, title):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=df, x='disease_category', y='total_cases', palette='viridis', ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Disease Category')
    ax.set_ylabel('Total Cases (Sum)')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    return fig

# To visualize the total cases of a specific disease category across different states
def plot_cases_by_state_for_category(df, category, title):
    filtered_df = df[df['disease_category'] == category].sort_values(by='total_cases', ascending=False)
    if filtered_df.empty:
        return None, f"No data for '{category}' to plot by state."

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.barplot(data=filtered_df, x='state', y='total_cases', palette='plasma', ax=ax)
    ax.set_title(title)
    ax.set_xlabel('State')
    ax.set_ylabel('Total Cases')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    return fig, None

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

