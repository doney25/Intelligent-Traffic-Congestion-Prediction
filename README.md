# 🚦 Traffic Intelligence System

An intelligent traffic analysis and decision-support system that uses historical traffic data, machine learning, and traffic engineering techniques to analyze congestion, forecast traffic conditions, estimate travel-time reliability, and recommend optimized traffic signal timings.

---

## 📌 Project Overview

Traffic congestion is a major challenge in urban transportation. Traditional traffic management systems often rely on fixed signal timings, which may not adapt effectively to changing traffic conditions throughout the day.

The **Traffic Intelligence System** aims to address this challenge by combining historical traffic analysis, machine learning-based forecasting, and traffic signal optimization techniques.

The project focuses on the **I-94 corridor between Minneapolis and St. Paul, Minnesota, USA**.

It uses historical traffic data to:

- Identify traffic congestion patterns.
- Forecast upcoming traffic conditions.
- Analyze peak traffic hours.
- Estimate travel-time reliability.
- Recommend improved traffic signal timings.
- Compare existing and optimized signal timing strategies.

> **Note:** This is a historical-data-based traffic intelligence prototype. It does not receive a live traffic feed or directly control real-world traffic signals.

---

## 🎯 Project Objectives

- Analyze historical traffic patterns and congestion levels.
- Identify peak traffic hours and recurring congestion trends.
- Forecast future traffic conditions using machine learning models.
- Estimate travel-time reliability across different departure times.
- Recommend optimized traffic signal timings based on estimated traffic demand.
- Compare baseline and optimized signal timing strategies.
- Present traffic insights through an interactive dashboard.

---

## 🖥️ Dashboard Features

The application consists of seven interactive modules.

### 1. Live Map

Provides a map-based visualization of the I-94 corridor.

**Features:**
- Interactive traffic corridor map.
- Traffic conditions represented using different colours.
- Green indicates flowing traffic.
- Orange/yellow indicates slow traffic.
- Red indicates congested traffic.
- Signalized intersections displayed as markers.
- Historical date and time controls.

### 2. Traffic Forecast

Uses machine learning models to forecast upcoming traffic conditions.

**Features:**
- One-hour and two-hour traffic forecasts.
- Historical traffic trends and predicted traffic values.
- Comparison of actual and predicted traffic.
- Visualization of forecast uncertainty.
- Traffic condition thresholds.

**Models evaluated:**

| Model | Description |
|---|---|
| Gradient Boosting | Tree-based machine learning model |
| GRU | Gated Recurrent Unit |
| LSTM | Long Short-Term Memory |
| GRU + LSTM Ensemble | Combination of GRU and LSTM predictions |
| Same-Hour-Last-Week | Historical baseline model |

### 3. Traffic Signals

Provides recommendations for adjusting traffic signal timings based on estimated traffic demand.

**Features:**
- Estimated traffic flow at selected intersections.
- Traffic load analysis.
- Recommended signal cycle lengths.
- Green-time allocation for different approaches.
- Comparison of signal timing strategies.

### 4. Peak Hours Analysis

Identifies recurring traffic patterns and congestion periods.

**Features:**
- Morning and evening peak-hour analysis.
- Weekday and weekend traffic comparisons.
- Hourly congestion trends.
- Day-of-week and hour-based traffic heatmaps.
- Identification of high-traffic periods.

### 5. Travel-Time Reliability

Analyzes how travel times vary across different departure times.

**Features:**
- Median travel-time analysis.
- Travel-time distribution.
- 5th and 95th percentile travel-time ranges.
- Identification of reliable departure windows.
- Historical travel-time observations.

### 6. Before / After Comparison

Compares a baseline signal timing plan with an optimized signal timing plan.

**Features:**
- Average vehicle delay comparison.
- Estimated vehicle-hours saved per day.
- Traffic volume analysis.
- Intersection-level comparison.
- Visualization of delay differences.

The results represent model-based estimates rather than measured real-world improvements.

### 7. Models and Notes

Provides information about machine learning models and their evaluation.

**Features:**
- Model performance comparison.
- MAE and RMSE evaluation metrics.
- One-hour and two-hour forecast comparisons.
- Actual versus predicted traffic visualization.
- Model-related notes and methodology.

---

## 🤖 Machine Learning Model Evaluation

The following are the model comparison results displayed in the dashboard.

| Model | Forecast Horizon | MAE | RMSE |
|---|---:|---:|---:|
| Gradient Boosting | 1 Hour | 206 | 297 |
| GRU + LSTM Ensemble | 1 Hour | 209 | 304 |
| GRU | 1 Hour | 216 | 307 |
| LSTM | 1 Hour | 240 | 345 |
| Same-Hour-Last-Week | 1 Hour | 344 | 659 |
| Gradient Boosting | 2 Hours | 269 | 396 |
| GRU + LSTM Ensemble | 2 Hours | 306 | 444 |
| GRU | 2 Hours | 317 | 445 |
| Same-Hour-Last-Week | 2 Hours | 344 | 659 |
| LSTM | 2 Hours | 360 | 530 |

### Evaluation Metrics

- **MAE (Mean Absolute Error):** Measures the average absolute difference between actual and predicted traffic values.
- **RMSE (Root Mean Squared Error):** Measures prediction error while giving greater weight to larger errors.

Lower MAE and RMSE values indicate better predictive performance.

Based on the displayed evaluation results, **Gradient Boosting achieved the lowest MAE and RMSE for both forecast horizons**.

---

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Dashboard | Streamlit |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Deep Learning | TensorFlow / Keras |
| Data Visualization | Plotly |
| Interactive Mapping | Leaflet |
| Map Tiles | OpenStreetMap |
| Development Tools | VS Code, Jupyter Notebook |
| Version Control | Git, GitHub |

---

## ⚙️ Traffic Signal Optimization

The Traffic Signals module demonstrates how traffic signal timings can be adjusted based on estimated traffic demand.

The optimization process considers:

1. Estimated traffic flow for different approaches.
2. Traffic demand and intersection load.
3. Signal cycle length.
4. Green-time allocation.
5. Comparison of baseline and optimized timing plans.

The system uses traffic engineering concepts, including Webster-style signal cycle calculations.

The recommendations are intended for analysis and demonstration. Real-world implementation would require accurate intersection-level traffic counts, lane configurations, turning movements, pedestrian requirements, and safety validation.

---

## 📊 Data and Methodology

The project uses historical traffic data associated with the I-94 corridor.

The general workflow is:

1. **Data Collection:** Obtain historical traffic observations.
2. **Data Preprocessing:** Clean and prepare traffic data.
3. **Exploratory Data Analysis:** Identify traffic patterns, trends, and congestion periods.
4. **Model Development:** Train and evaluate machine learning models.
5. **Traffic Forecasting:** Generate one-hour and two-hour traffic predictions.
6. **Signal Optimization:** Estimate traffic demand and recommend signal timing adjustments.
7. **Visualization:** Present results through interactive dashboard modules.

**Data limitation:** The available historical dataset ends in September 2018. Therefore, the system should be interpreted as a historical-data replay and forecasting prototype, not a live traffic monitoring platform.

---
