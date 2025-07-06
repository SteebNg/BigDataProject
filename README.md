# 🦠🩺 Healthcare Data Insights: Disease Trends in Malaysia

This project is an interactive Streamlit dashboard designed for the **5011CEM Big Data Programming Project** module. It focuses on analyzing and predicting disease trends in Malaysia using a provided dataset.

## Project Overview

The dashboard offers a user-friendly interface to:
* **Analyze Historical Disease Trends:** Visualize total cases and incidence rates of various disease categories over time (2017-2021).
* **Identify Geographical Hotspots:** Pinpoint states with higher disease burdens for specific categories.
* **Forecast Future Trends:** Utilize a Machine Learning model (Prophet or Linear Regression) to predict future disease cases for selected categories.
* **Explore Raw Data:** Provide an interface to view the raw and processed datasets.

## Features

* **Interactive Dashboard:** Built with Streamlit for dynamic data exploration.
* **Disease Categorization:** Groups similar diseases into broader categories for macro-level analysis.
* **Time Series Analysis:** Visualizes temporal patterns of disease spread.
* **Geographical Analysis:** Highlights state-wise distribution of cases.
* **Predictive Analytics:** Integrates a machine learning model for forecasting.

## Technologies Used

* **Python:** Core programming language.
* **Streamlit:** For creating the interactive web application.
* **Pandas:** For data manipulation and analysis.
* **Matplotlib & Seaborn:** For static data visualizations.
* **Scikit-learn:** (If used, e.g., for Linear Regression) For general machine learning algorithms.

## Setup and Installation

Follow these steps to get the project up and running on your local machine.

### 1. Prerequisites

* **Python** installed on your system. You can download it from [python.org](https://www.python.org/downloads/).
* **pip** (Python package installer), which usually comes with Python.
* **sts_state.csv** and **INTI_Logo.png** installed in the same location as the Python file.

### 2. Clone the Repository (Optional, if you're managing files manually)

If you're using Git, you can clone this repository:

```bash
git clone <your-repository-url>
cd <your-repository-name>
