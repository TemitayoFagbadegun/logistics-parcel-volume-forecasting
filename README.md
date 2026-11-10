# Logistics Parcel Volume Forecasting

A machine learning project that forecasts daily parcel volumes for a simulated logistics operation. The goal is to explore how historical demand and calendar features can support operational planning such as staffing, sorting capacity and vehicle allocation.

> **Note:** This is an independent portfolio project using fully synthetic data. It does not contain DHL data and is not affiliated with or endorsed by DHL.

## Project objective

Predict the number of parcels a depot is likely to process on a future day using information available from previous days and the calendar.

The project currently uses features including:

- previous day's parcel volume (`lag_1`)
- parcel volume from the same day one week earlier (`lag_7`)
- day of week
- month
- bank-holiday indicator
- Black Friday period indicator
- Christmas peak indicator

## Current results

The first stage compares a simple operational baseline with a Linear Regression model on the same unseen 2026 test period.

| Model | Test MAE | Improvement vs baseline |
|---|---:|---:|
| 7-day lag baseline | 1,182.84 | — |
| Linear Regression | **1,029.12** | **~13.0%** |

**MAE (Mean Absolute Error)** measures the average absolute difference between predicted and actual parcel volumes. Lower is better.

At this stage, Linear Regression reduces the average forecasting error by roughly 154 parcels per depot-day compared with simply using the same day from the previous week.

## Why this problem matters

Accurate parcel-volume forecasts could help a logistics operation make better decisions around:

- workforce and shift planning
- sorting capacity
- vehicle and route preparation
- overtime requirements
- peak-period readiness

The project is intentionally framed around an operational problem rather than only model accuracy.

## Repository structure

```text
logistics-parcel-volume-forecasting/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── README.md
├── notebooks/
│   └── 01_baseline_and_linear_regression.ipynb
└── src/
    └── generate_data.py
```

## Dataset

The dataset is generated locally by `src/generate_data.py` and contains four fictional depots from January 2024 to July 2026.

Example fields include:

- `date`
- `depot`
- `temperature_c`
- `rainfall_mm`
- `is_bank_holiday`
- `is_black_friday_period`
- `is_christmas_peak`
- `parcel_volume`

Synthetic demand patterns include weekly seasonality, peak periods, holidays, weather variation and a gradual long-term trend.

## Methodology so far

### 1. Exploratory data analysis

The initial analysis investigates questions such as:

- Which depot processes the most parcels on average?
- Which day of the week is busiest?
- How does Christmas peak demand compare with normal periods?
- Are weekends quieter than weekdays?
- Is there a longer-term growth trend?

### 2. Baseline forecast

A simple baseline predicts today's parcel volume using the volume from the same depot seven days earlier:

```python
prediction = lag_7
```

This provides a minimum benchmark that more advanced models should beat.

### 3. Feature engineering

Historical and calendar features are created, including `lag_1`, `lag_7`, month and day-of-week information.

### 4. Time-based train/test split

Because this is a forecasting problem, the data is not randomly shuffled. Data before **1 January 2026** is used for training and data from **1 January 2026 onward** is used for testing.

This gives:

- Training rows: **2,896**
- Testing rows: **848**

### 5. Linear Regression

Linear Regression is trained as the first machine-learning benchmark. Its MAE of **1,029.12** beats the equivalent test-period baseline MAE of **1,182.84**.

## Running the project

Clone the repository and install the dependencies:

```bash
pip install -r requirements.txt
```

Generate the synthetic dataset:

```bash
python src/generate_data.py
```

Then open:

```text
notebooks/01_baseline_and_linear_regression.ipynb
```

Run the notebook from top to bottom.

## Planned improvements

The next stages of the project will include:

- Random Forest regression
- Gradient Boosting
- additional rolling and lag features
- model comparison on a consistent test period
- feature importance / model explainability
- visual comparison of predictions vs actual demand
- a small Streamlit application for interactive forecasts

## Skills demonstrated

Python · Pandas · NumPy · Matplotlib · scikit-learn · Exploratory Data Analysis · Feature Engineering · Regression · Time-Series-Aware Validation · Model Evaluation
