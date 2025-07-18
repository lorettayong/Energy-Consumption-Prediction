# Project: PJME Hourly Energy Consumption

This project focuses on building a robust machine learning model to predict hourly electricity demand for the PJM Interconnection, a major wholesale electricity market in the United States. Accurate energy forecasting is critical for optimising power generation, managing grid stability, planning resource allocation, and ensuring efficient energy market operations.

Using historical data, this project aims to identify key drivers of consumption patterns, including temporal features and external factors such as temperature, and ultimately deploy machine learning models to provide reliable future demand predictions. The final model demonstrates strong capabilities in capturing complex seasonal patterns and trends in energy consumption time series.

# Motivation

The ability to accurately forecast energy demande is a cornerstone of modern energy management. Underestimating demand can lead to costly power outages and economic losses, while overestimating it results in wasted resources and increased operational costs. With the increasing integration of intermittent renewable energy sources and the volatility introduced by climate change, the need for precise and agile forecasting models has never been greater.

I am undertaking this project to deepen my practical skills in time series analysis, master advanced feature engineering techniques for temporal data, and apply powerful machine learning algorithms to a real-world problem with significant societal and economic impact. This endeavour allows me to explore how data science can contribute to more sustainable and efficient energy systems.

# Dataset

The primary dataset used in this project is the **PJME Hourly Energy Consumption**. It comprises historical hourly electricity demand (measured in Megawatts - MW) for the PJM Interconnection, covering the period of 01 January 2002, 01:00 am to 03 August 2018, 12:00 am.

* **Source: [Kaggle - PJME Hourly Energy Consumption](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption)

# Project Structure
- `energy-prediction.ipynb`: Jupyter Notebook containing the data loading, exploratory data analysis (EDA), model building, and evaluation steps.
- `requirements.txt`: Lists all the Python libraries and their versions required to run this project.

# How to Run the Project
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/lorettayong/energy-consumption-prediction.git
   cd energy-consumption-prediction
   ```

2. **Create and Activate a Virtual Environment:**
   ```bash
   python -m venv venv
   # On Windows: .\venv\Scripts\activate
   # On macOS/Linux: source venv/bin/activate
   ```
3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *Generate this file from your active `venv` after installing all libraries: `pip freeze > requirements.txt`)*

4. **Download the Dataset:** Download the `PJME_hourly.csv` file from the [Kaggle - PJME Hourly Energy Consumption](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption) and place it into the `data/` directory within the cloned repository.
  
5. **Run Jupyter Notebook:**
   ```bash
   jupyter lab
   ```
   Open the `energy_prediction_pipeline.ipynb` notebook and run all cells sequentially to execute the entire machine learning pipeline from data loading to model evaluation.

# Initial Data Overview

This dataset provides a long, high-frequency time series ideal for identifying seasonal patterns (daily, weekly, annual) and long-term trends essential for comprehensive energy forecasting.

* **Columns:**
  * `Datetime`: Timestamp of the hourly record
  * `PJME_MW`: Actual hourly electricity consumption in MW.

# Project Phases

## Phase 1: Data Preprocessing and Exploratory Data Analysis (EDA)
* **Objectives:** Load the dataset, understand its structure and content, handle initial data quality issues, convert the timestamp to the correct format, and perform some initial visualisations to gain insights to the energy consumption patterns.
* **Key Activities:**
  * Loaded `PJME_hourly.csv` file into a Pandas Dataframe.
  * Conducted initial checks using `df.head()`, `df.info()`, and `df.describe()` to understand the data structure, types, and basic statistics.
  * Converted the `Datetime` column from an object (string) type to a proper `datetime` object, set as the DataFrame's index, which is crucial for time series operations, and sorted the index in chronological order to ensure data integrity.
  * Identified four instances of duplicate timestamps and handled them by keeping the first occurrence to maintain uniqueness in the time series index.
  * Generated a complete, continuous hourly `DatetimeIndex` that spans the entire data range, and `re-index`ed the DataFrame to this full range by introducing explicit `NaN` values for any previously missing hours. This is needed as the dataset exhibited instances of skipped hours (e.g. 2-hour gaps), primarily due to Daylight Saving Time (DST) changes.
  * Interpolated the `NaN` values that were introduced through reindexing (to represent the skipped hours) using a linear method. This ensures a perfectly continuous and regularly spaced hourly time series, which is a prerequisite for many time series modelling techniques.
  * Plotted the `PJME_MW` consumption over the entire period and in sections of the time series (in specific month and week) to visualise long-term trends and observe granular weekly and daily seasonality more clearly.
* **Observations from EDA:**
  * Strong annual seasonality: Energy demand shows clear yearly cycles, typically peaking during summer (for air conditioning purposes) and winter (for heating purposes), with lower demands in spring and autumn.
  * Distinct weekly seasonality: Weekday consumption patterns differ significantly from weekends, with lower demands usually on Saturdays and Sundays.
  * Clear daily seasonality: Within each 24-hour cycle, consumption follows a predictable pattern, often with peaks in the mornings and evenings, and troughs in the very early hours of the morning.
  * The overall yearly trend appears relatively stable, although specific years might show minor variations.

## Phase 2: Feature Engineering
* **Objective:** Create rich predictive features from the available energy consumption data and integrate critical external information.
* **Key Activities:**
  * Extracted components from the `DatetimeIndex` such as `hour_of_day`, `day_of_week`, `week_of_year`, `month`, `quarter`, `year`, and `is_weekend` (a binary flag).
  * Generated lagged features, including `lag_1_hour`, `lag_24_hour`, `lag_168_hour`, which represent past consumption values and are highly predictive of future demand.
  * Integrated features indicating US federal public holidays, which typically exhibit distinct energy consumption patterns due to reduction in business and industrial activities.
  * Sourced and incorporated historical temperature data corresponding to a selected location within the PJM region, Chicago. These include current temperatures, potentially lagged temperatures, and various temperature-derived features such as rolling averages and squared temperatures, given the strong correlation between temperature and demand for energy.

# Next Steps (Future Work)
* ~~**Feature Engineering:** Create new predictive features from the existing `DatetimeIndex` of the energy consumption data, including the extraction of components such as hour of the day, day of the week, month, quarter, year, and the creation of flags for weekends.~~
* ~~**External Data Integration:** Source and incorporate historical temperature data corresponding to the PJM region to demonstrate the correlation between temperature and energy demand, and integrate features indicating national public holidays to exhibit distinct patterns of energy consumption.~~
* **Model Building:** Select suitable regression models for time series forecasting, such as XGBoost Regressor, LightGBM Regressor, and Facebook Prophet, train these models on the prepared training data, and conduct hyperparameter tuning using techniques such as `GridSearchCV` and `RandomizedSearchCV` combined with a time series-aware cross-validation strategy.
* **Model Evaluation:** Perform an initial evaluation of their performance using metrics like Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and R-squared ($R^2$) on the test set.
* **Model Deployment:** Develop a simple and interactive web application to showcase the practical accessibility of the trained forecasting model.