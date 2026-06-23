# Metro Interstate Traffic Volume Prediction: Ensemble Methods Comparison

## Overview
This repository contains a Jupyter Notebook (`XGBoost_RandomForest_ADABoost.ipynb`) dedicated to predicting the hourly traffic volume on Interstate 94 (I-94) using a combination of weather data and temporal features. 

The primary objective of this project is to implement, evaluate, and compare three popular ensemble machine learning algorithms:
* **Random Forest Regressor**
* **AdaBoost Regressor**
* **XGBoost Regressor**

## Dataset
**Name:** Metro Interstate Traffic Volume
The dataset contains hourly traffic volume alongside weather conditions (temperature, rain, snow, cloud cover, and general weather descriptions) and datetime information.

**Target Variable:**
* `traffic_volume`: The numeric count of hourly traffic.

**Key Features Engineered:**
* `hour`, `day_of_week`, `month`, `year`: Extracted from the raw `date_time` column to capture diurnal and seasonal trends.
* `is_weekend`: Binary indicator for weekends.
* `is_holiday`: Binary indicator derived from the original `holiday` feature.
* `weather_main_enc`: Label-encoded categorical weather descriptions.

## Notebook Structure
The analysis is structured into the following key sections:

1.  **Imports & Configuration:** Loading necessary Python libraries (`pandas`, `numpy`, `scikit-learn`, `xgboost`, `matplotlib`, `seaborn`).
2.  **Load & Inspect the Dataset:** Initial data exploration, checking data types, basic statistics, and handling missing values.
3.  **Exploratory Data Analysis (EDA):** Visualizing the distribution of traffic volume, analyzing traffic patterns by hour of the day and day of the week, and assessing the impact of different weather conditions on traffic.
4.  **Data Preprocessing:** Cleaning anomalous data (e.g., $0$ K temperatures), label encoding, feature extraction, and splitting the data into training (80%) and testing (20%) sets.
5.  **Model Training & Evaluation:** Fitting the three ensemble models and evaluating them using MAE, MSE, RMSE, and $R^2$ score.
6.  **Appendix (Educational Resource):** A highly detailed breakdown of the machine learning concepts, evaluation metrics, and the internal workings of Bagging (Random Forest) and Boosting (AdaBoost, XGBoost) algorithms.

## Key Findings & Model Performance
The models were evaluated on the 20% hold-out test set. Both XGBoost and Random Forest demonstrated excellent predictive capabilities, significantly outperforming AdaBoost on this dataset.

| Model | MAE | RMSE | $R^2$ Score |
| :--- | :--- | :--- | :--- |
| **XGBoost** | ~ 238.80 | ~ 397.60 | **0.9605** |
| **Random Forest** | ~ 245.57 | ~ 425.95 | **0.9546** |
| **AdaBoost** | ~ 620.11 | ~ 835.30 | 0.8255 |

*Note: XGBoost performed best due to its deeper tree configuration (`max_depth=8`) and gradient-based optimization, allowing it to capture complex non-linear interactions (like distinct rush-hour patterns on weekdays vs. weekends) better than the shallow decision stumps used by AdaBoost.*

## Requirements
To run this notebook, ensure you have the following Python packages installed:
* `numpy`
* `pandas`
* `matplotlib`
* `seaborn`
* `scikit-learn`
* `xgboost`

## Usage
1.  Ensure `Metro_Interstate_Traffic_Volume.csv` is located in the same directory as the notebook.
2.  Open `XGBoost_RandomForest_ADABoost.ipynb` using Jupyter Notebook or JupyterLab.
3.  Execute the cells sequentially to reproduce the data processing, visualizations, and model evaluations.
