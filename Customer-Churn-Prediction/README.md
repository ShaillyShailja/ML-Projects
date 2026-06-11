# Bank Customer Churn Prediction

## Overview

This project focuses on predicting **customer churn** for a bank using machine learning techniques.
Customer churn refers to customers leaving the bank, which is a critical business problem as retaining customers is more cost-effective than acquiring new ones.

The objective is to build and evaluate classification models that can accurately identify customers who are likely to churn.

---

## Problem Statement

Given customer-related features such as demographics, account details, and transaction behavior, the goal is to:

* Predict whether a customer will **churn (1)** or **not churn (0)**
* Improve the bank's ability to take **proactive retention actions**

---

## Dataset Description

The dataset contains customer-level information including:

* Customer demographics (age, gender, geography)
* Account information (balance, tenure, products)
* Activity metrics (transactions, engagement)
* Target variable: **Churn (Exited)**

---

## Approach

### 1. Data Preprocessing

* Handling missing values (if any)
* Encoding categorical variables
* Feature scaling for model performance
* Train-test split

---

### 2. Models Used

#### Logistic Regression (Baseline)

* Simple and interpretable model
* Used as a benchmark for comparison

#### Multilayer Perceptron (MLP)

* Feedforward neural network
* Captures non-linear relationships in data

---

## Evaluation Metrics

Since churn prediction is a **classification problem with class imbalance**, the following metrics were used:

* **Accuracy**
* **Precision**
* **Recall (Primary Metric)**
* **F1-Score**

### Why Recall is Important?

Recall measures how many actual churn customers are correctly identified.

> Missing a churned customer is costly → hence **high recall is prioritized**

---

## Results & Insights

| Model               | Accuracy | Recall      | F1-Score |
| ------------------- | -------- | ----------- | -------- |
| Logistic Regression | Moderate | Low (~0.44) | Moderate |
| MLP Classifier      | Similar  | High (~1.0) | Higher   |

### Key Findings:

* MLP significantly improves **recall**, capturing almost all churn cases
* Logistic Regression fails to detect many churn customers
* Slight trade-off in precision may occur, but recall improvement is more valuable

---

## Visualizations

* Model performance comparison (bar charts)
* Confusion matrix
* Feature distributions

---

## Tech Stack

* Python
* Libraries:

  * NumPy
  * Pandas
  * Scikit-learn
  * Matplotlib / Seaborn

---

## How to Run

1. Clone the repository:

```bash
git clone https://github.com/your-username/repo-name.git
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the notebook:

* Open `.ipynb` file in Jupyter Notebook or Jupyter Lab

---

## Conclusion

* Neural networks (MLP) outperform traditional models for this dataset
* Recall is the most critical metric in churn prediction
* The model can help banks **reduce customer attrition** through early intervention

---

## Future Improvements

* Hyperparameter tuning for MLP
* Use advanced models (XGBoost, Random Forest)
* Handle class imbalance (SMOTE, class weights)
* Deploy model using Flask / Streamlit

---

## Author

**Shailly Shailja**
Audit & Assurance Analytics Specialist Senior | Deloitte USI, Hyderabad

GitHub: https://github.com/ShaillyShailja
---
