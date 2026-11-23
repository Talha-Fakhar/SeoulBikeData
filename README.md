# Seoul Bike Sharing Demand Forecasting & Explainable AI

An end-to-end machine learning and visual analytics project predicting hourly bike rental demand (`rented_bike_count`) across Seoul using meteorological, seasonal, and temporal features.

## Project Overview
Urban bike-sharing systems face frequent station stockouts during peak commuting hours and underutilization during adverse weather. This project builds a leak-free forecasting pipeline to predict hourly rental demand and uses Explainable AI (SHAP) to uncover actionable insights for fleet rebalancing and maintenance scheduling.

## Key Features & Methodology
* **Data Wrangling & Outlier Strategy:**
  * Cleaned 8,760 hourly records, handled missing values, and standardized `DD/MM/YYYY` timestamps.
  * Applied domain-informed outlier handling: preserved genuine rush-hour demand peaks and extreme temperatures, capped heavily skewed precipitation (`rainfall`, `snowfall`) and `wind_speed`/`radiation` at the 99th percentile, and enforced physical sensor bounds.
* **Feature Engineering:**
  * Extracted calendar and cyclical temporal signals (`hour`, `day`, `month`, `weekday_num`, `is_weekend`) and engineered lag features while preventing target leakage.
* **EDA & Interactive Visual Analytics:**
  * Analyzed seasonal trends, bimodal weekday commuter peaks vs. weekend leisure curves, and weather correlations using Matplotlib, Seaborn, and interactive **Plotly** visualizations.
* **Time-Aware Modeling & Hyperparameter Tuning:**
  * Used a strict chronological `80/10/10` Train/Validation/Test split alongside `TimeSeriesSplit` cross-validation.
  * Benchmarked a Naive Mean Baseline, Linear Regression, Random Forest, and **Tuned XGBoost** (`RandomizedSearchCV` with L1/L2 regularization and `colsample_bytree` to prevent lag dominance).
* **Explainable AI (XAI):**
  * Applied **SHAP** (Global Beeswarm, Global Feature Importance Bar, and Local Waterfall plots) to verify directional weather/time impacts and explain individual peak-hour forecasts.

## Model Performance Summary

| Model | Test RMSE | Test MAE | Notes |
| :--- | :---: | :---: | :--- |
| **Naive Mean Baseline** | `480.68` | — | Constant training mean benchmark |
| **Linear Regression** | `233.57` | — | Linear baseline (~51% RMSE reduction) |
| **Random Forest (Tuned)** | `127.69` | `81.64` | Captures non-linear hourly/weather splits |
| **Tune-XGBoost (Selected)** | **`108.86`** | **`72.61`** | Best generalization & smallest train-val gap |

## Tech Stack
* **Data Manipulation:** Python, Pandas, NumPy
* **Visualization:** Plotly, Matplotlib, Seaborn
* **Machine Learning:** Scikit-Learn, XGBoost
* **Explainability:** SHAP

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/Talha-Fakhar>/SeoulBikeData.git
   cd SeoulBikeData
