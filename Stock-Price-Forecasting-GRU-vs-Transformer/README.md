# Stock Market Analysis: GRU vs Transformer for Time Series Forecasting

## Overview

This project compares the performance of Recurrent Neural Networks (GRU) and Transformer architectures for stock price forecasting using historical stock market data.

The objective is to evaluate how well traditional sequence models and attention-based models can predict future stock closing prices and to analyze their strengths, weaknesses, and forecasting accuracy.

---

## Dataset

**Dataset:** Historical Stock Market Data

**File Used:** `CEF.csv`

**Features Used:**
- Date
- Close Price

The project focuses on univariate time-series forecasting using the stock's closing price.

---

## Project Workflow

### 1. Data Loading and Exploration

- Load historical stock market data
- Convert dates into datetime format
- Visualize closing prices
- Identify trends and seasonality
- Perform stationarity testing using:
  - Augmented Dickey-Fuller (ADF) Test
  - KPSS Test

---

### 2. Data Preprocessing

The following preprocessing steps were performed:

- Handle missing values using forward fill
- Remove remaining null values
- Temporal train-test split (85%-15%)
- Min-Max normalization using training data only
- Sequence generation for supervised learning

---

### 3. GRU Model Implementation

A stacked GRU architecture was implemented using TensorFlow/Keras.

#### Architecture

- 2 GRU layers
- 64 hidden units
- Dropout Regularization (0.2)
- Dense Output Layer

#### Training Configuration

| Parameter | Value |
|------------|---------|
| Optimizer | Adam |
| Loss Function | Mean Squared Error |
| Epochs | 50 |
| Batch Size | 32 |

---

### 4. Transformer Model Implementation

A Transformer Encoder architecture was implemented using TensorFlow/Keras.

#### Key Components

- Dense Input Projection
- Sinusoidal Positional Encoding
- Multi-Head Self-Attention
- Feed Forward Network
- Residual Connections
- Layer Normalization
- Global Average Pooling

#### Architecture

| Parameter | Value |
|------------|---------|
| Encoder Layers | 2 |
| Attention Heads | 4 |
| Model Dimension (d_model) | 64 |
| Feed Forward Dimension | 128 |

---

### 5. Model Evaluation

Both models were evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Percentage Error (MAPE)
- R² Score

---

### 6. Visualizations

The notebook generates:

- Stock Price Trend
- Seasonality Analysis
- Training Loss Curves
- Actual vs Predicted Prices
- Residual Analysis
- Model Performance Comparison
- Training Curve Comparison

---

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-Learn
- TensorFlow / Keras
- Statsmodels

---

## Repository Structure

```text
Stock_Market_Analysis/
│
├── Stock_Market_Analysis.ipynb
├── CEF.csv
├── README.md
└── outputs/
    ├── loss_curves.png
    ├── predictions.png
    └── comparison_plots.png
```

---

## Results

The project compares:

- Forecasting accuracy
- Training time
- Model complexity
- Number of parameters
- Generalization capability

The final comparison highlights the trade-offs between recurrent architectures and attention-based architectures for financial time-series forecasting.

---

## Future Improvements

- Multi-step forecasting
- Hyperparameter tuning
- Additional technical indicators
- LSTM vs GRU comparison
- Cross-validation techniques for time series
- Incorporation of market sentiment data
- Advanced Transformer architectures (Informer, Time Series Transformer)

---

## Author

**Shailly Shailja**
Audit & Assurance Analytics Specialist Senior | Deloitte USI, Hyderabad

GitHub: https://github.com/ShaillyShailja
