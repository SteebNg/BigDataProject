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

