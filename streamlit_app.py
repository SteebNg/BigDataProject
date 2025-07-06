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

    # --- Dashboard Overview Section ---
    if selected_analysis == "Dashboard Overview":
        st.header("Dashboard Overview: Key Insights at a Glance")
        st.write("A summary of the most critical findings from the disease data.")

        # Display KPIs
        total_cases = processed_df['cases'].sum()
        num_diseases = processed_df['disease'].nunique()
        num_categories = processed_df['disease_category'].nunique()
        num_states = processed_df['state'].nunique()
        reporting_years = processed_df['year'].unique()
        min_year, max_year = min(reporting_years), max(reporting_years)

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric(label="Total Recorded Cases (2017-2021)", value=f"{total_cases:,}")
        with col2:
            st.metric(label="Unique Diseases Tracked", value=num_diseases)
        with col3:
            st.metric(label="Disease Categories", value=num_categories)
        with col4:
            st.metric(label="States Covered", value=num_states)
        with col5:
            st.metric(label="Analysis Period", value=f"{min_year}-{max_year}")

        st.markdown("---")

        # Top Disease Categories Bar Chart
        st.subheader("Overall Top Disease Categories by Total Cases")
        fig_overall_cases = plot_overall_cases_by_category(overall_category_summary,
                                                           'Overall Total Cases by Disease Category')
        st.pyplot(fig_overall_cases)
        st.markdown(f"""
            <p style='font-size: small; text-align: center;'>
            The chart above illustrates the aggregated number of cases for each disease category across all states and years.
            It helps to quickly identify which disease categories have historically had the highest burden in Malaysia.
            </p>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Cases Over Time for Top Categories (Interactive)
        st.subheader("Disease Cases Over Time")
        st.write("Select disease categories to compare their total cases over the years.")
        unique_categories = sorted(processed_df['disease_category'].unique())
        selected_categories_time = st.multiselect(
            "Select Disease Categories for Time Trend:",
            options=unique_categories,
            default=unique_categories[0] if unique_categories else []  # Default to first category if available
        )

        if selected_categories_time:
            filtered_time_df = yearly_category_trends[
                yearly_category_trends['disease_category'].isin(selected_categories_time)]
            fig_cases_time = plot_cases_over_time(filtered_time_df,
                                                  'Selected Disease Categories: Total Cases Over Time')
            st.pyplot(fig_cases_time)
            st.markdown(f"""
                <p style='font-size: small; text-align: center;'>
                This graph shows the trend of total cases for the selected disease categories over the years.
                Observe if cases are increasing, decreasing, or remaining stable.
                </p>
                """, unsafe_allow_html=True)
        else:
            st.info("Please select at least one disease category to view its trend over time.")

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

