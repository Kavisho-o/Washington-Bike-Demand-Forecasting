# Washington Bike Demand Forecasting

## 📌 Overview

This project predicts daily bike rental demand using weather,
seasonal, and calendar-based features.

The goal is to compare linear and tree-based models and build
a production-ready regression pipeline.

---

## 📊 Dataset

Bike sharing demand dataset containing:

- Weather conditions
- Seasonal information
- Holiday indicators
- Temporal features
- Rental counts (target variable)

---

## 🛠 Feature Engineering

- Cyclical encoding for month and weekday
- One-hot encoding for categorical features
- Standard scaling for numerical features

---

## 🤖 Models Compared

- Linear Regression
- Ridge & Lasso
- Decision Tree
- Random Forest
- Gradient Boosting

Final selected model:
**Tuned Gradient Boosting Regressor**

---

## 📈 Final Performance

- Test RMSE ≈ 635
- Test R² ≈ 0.899

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
python main.py