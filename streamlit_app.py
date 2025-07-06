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

elif selected_analysis == "Geographical Analysis":
        st.header("Geographical Analysis: High-Risk States")
        st.write("Identify states with higher total cases or incidence for specific disease categories.")

        unique_categories_geo = sorted(state_category_trends['disease_category'].unique())
        selected_category_geo = st.selectbox(
            "Select Disease Category for State-wise Analysis:",
            options=unique_categories_geo
        )

        if selected_category_geo:
            fig_state_cases, msg = plot_cases_by_state_for_category(state_category_trends, selected_category_geo,
                                                                   f'Total {selected_category_geo} Cases by State')
            if fig_state_cases:
                st.pyplot(fig_state_cases)
                st.markdown(f"""
                <p style='font-size: small; text-align: center;'>
                This bar chart displays the total cases of **{selected_category_geo}** in each Malaysian state.
                States with taller bars indicate higher cumulative cases for this category, suggesting areas for targeted intervention.
                </p>
                """, unsafe_allow_html=True)
            else:
                st.info(msg)

# --- Data Explorer Section ---
    elif selected_analysis == "Data Explorer":
        st.header("Data Explorer")
        st.write("Browse the raw and processed datasets.")

        data_view_option = st.radio(
            "Select Data View:",
            ("Raw Data", "Processed Data")
        )

        if data_view_option == "Raw Data":
            st.subheader("Raw Dataset (`std_state.csv`)")
            st.dataframe(df_raw)
            with st.expander("Show Raw Data Info"):
                buffer = pd.io.common.StringIO()
                df_raw.info(buf=buffer)
                st.text(buffer.getvalue())
        else:
            st.subheader("Processed Dataset (with Categories and Year)")
            st.dataframe(processed_df)
            with st.expander("Show Processed Data Info"):
                buffer = pd.io.common.StringIO()
                processed_df.info(buf=buffer)
                st.text(buffer.getvalue())
            with st.expander("Unique Disease Categories and Counts"):
                st.dataframe(processed_df['disease_category'].value_counts())

    # --- About This Project Section ---
    elif selected_analysis == "About This Project":
        st.header("About This Project")
        st.markdown("""
        This project is part of the **5011CEM Big Data Programming Project** module at INTI International College Penang,
        in collaboration with Coventry University, UK.

        ### Project Objective
        To apply big data analysis and programming techniques to a realistic healthcare scenario in Malaysia,
        specifically focusing on **Disease Trend Analysis and Prediction**.

        ### Key Features
        * **Data Loading & Pre-processing:** Handles raw `.csv` data, converts date formats, and applies custom disease categorization.
        * **Disease Categorization:** Groups similar diseases into broader categories for macro-level analysis.
            *(Note: The current categorization is illustrative and should be thoroughly researched and justified in your assignment report.)*
        * **Trend Analysis:** Visualizes historical trends of disease cases and incidence rates over time (2017-2021).
        * **Geographical Hotspot Identification:** Identifies states with higher disease burdens for specific categories.
        * **Interactive Dashboard:** Provides a user-friendly interface to explore data and visualizations.
        * **Predictive Analysis (Machine Learning):** Integrates a machine learning model to forecast future disease cases.
            * **Chosen Algorithm:** We utilize the **Prophet** library (developed by Meta/Facebook) for its robust capabilities in time series forecasting, especially for handling trends and potential seasonality in future, more granular datasets. *(If you use Linear Regression, change this to: "We utilize **Linear Regression** for its simplicity and interpretability as a baseline forecasting model.")*
            * **Forecasting Goal:** To predict the total cases for selected disease categories for upcoming years.

        ### Technologies Used
        * **Python:** The core programming language.
        * **Pandas:** For efficient data manipulation and analysis.
        * **Matplotlib & Seaborn:** For static and informative data visualization.
        * **Streamlit:** For building the interactive web application/dashboard.
        * **Prophet (Meta/Facebook):** For time series forecasting. *(Remove this line if you are only using Linear Regression)*
        * **Scikit-learn:** For machine learning algorithms, specifically Linear Regression. *(Add this line if you are using Linear Regression, or keep both if you compare them)*

        ### Future Enhancements & Key Discussion Points for Report
        * **Integration of Additional Data:** Incorporate demographic data (e.g., age groups, gender distribution, population density) or environmental factors for more granular risk group analysis and potentially more accurate predictions.
        * **Advanced Geospatial Visualizations:** Explore more sophisticated mapping techniques to display geographical patterns.
        * **Model Evaluation & Comparison:** Implement metrics like RMSE, MAE, or R-squared to objectively evaluate the performance of the predictive model. If multiple models are considered (e.g., Prophet vs. Linear Regression), a comparative analysis would be valuable.
        * **Hyperparameter Tuning:** Investigate and discuss how tuning model parameters (e.g., `changepoint_prior_scale` in Prophet) can impact forecasting accuracy.
        * **Real-time Data Integration:** Discuss the challenges and possibilities of connecting to larger, real-time healthcare datasets.
        * **Ethical Considerations:** Reflect on the ethical implications of using predictive analytics in healthcare, including data privacy, bias in predictions, and responsible communication of forecasts.

        ### Assignment Guidance Reminder
        Remember to detail all aspects of this project in your assignment report, including:
        * Problem Definition & Literature Review
        * Comprehensive Data Analysis (Exploratory Data Analysis, detailed explanation of Machine Learning Algorithms applied, Algorithm Complexity, Model Evaluation, and Hyperparameter Tuning)
        * Professional Practices (Version control using Git/GitHub, ethical considerations, project management)
        * Clear Interpretation of Results from both historical analysis and predictive models
        * Appropriate Diagrams (Data Flow Diagram, Entity-Relationship Diagram, Flowcharts, UML diagrams, Gantt chart for project timeline)
        * A critical reflection on your work during the VIVA, highlighting challenges faced and lessons learned.
        """)
        st.markdown("---")
        st.write("Developed for the 5011CEM Big Data Programming Project.")
        st.write(f"Current Date: {pd.to_datetime('today').strftime('%Y-%m-%d')}")


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