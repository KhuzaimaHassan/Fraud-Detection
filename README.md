# Fraud Detection System

This project implements a machine learning-based system for detecting fraudulent financial transactions. It combines multiple datasets, applies advanced feature engineering, and trains various models to detect fraudulent transactions with high accuracy.

## Project Structure

```
fraud_detection_project/
├── Datasets/                  # Directory containing datasets
│   ├── Fraud.csv             # Main dataset
│   └── FraudData_sampled.csv # Sampled dataset
├── fraud_detection.py        # Main script for preprocessing and model training
├── app.py                    # Streamlit web application
├── setup.py                  # Setup script to organize project structure
├── models/                   # Directory for saved models
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   ├── lightgbm.pkl
│   ├── scaler.pkl
│   └── label_encoder.pkl
├── results/                  # Directory for metrics and visualizations
│   ├── model_comparison.csv
│   ├── confusion_matrix_*.png
│   └── *_comparison.png
└── explanations/            # Directory for model explanations
    └── shap_summary.png
```

## Dataset Description

The project uses two datasets:

### Fraud.csv
- **step**: Transaction time (hour or day)
- **type**: Transaction type (PAYMENT, TRANSFER, CASH_OUT, DEBIT)
- **amount**: Transaction amount
- **nameOrig**: Origin account ID
- **oldbalanceOrg**: Origin account balance before transaction
- **newbalanceOrig**: Origin account balance after transaction
- **nameDest**: Destination account ID
- **oldbalanceDest**: Destination account balance before transaction
- **newbalanceDest**: Destination account balance after transaction
- **isFraud**: Target variable (1 for fraudulent transactions)
- **isFlaggedFraud**: Flag for manually detected fraud

### FraudData_sampled.csv
A smaller sampled dataset with similar columns but without account ID columns.

## Feature Engineering

The project implements several engineered features to improve fraud detection:
- **transaction_diff**: Difference between old and new origin balance
- **dest_diff**: Difference between new and old destination balance
- **is_zero_balance**: Flag for zero balance transactions
- **amount_balance_ratio**: Ratio of transaction amount to origin balance
- **balance_change_ratio**: Ratio of balance change to original balance

## How to Run

### Step 1: Install Requirements

```bash
pip install -r requirements.txt
```

### Step 2: Train Models

```bash
python fraud_detection.py
```

This script will:
1. Load and combine both datasets
2. Apply feature engineering
3. Preprocess the data (handling categorical variables, scaling)
4. Balance the classes using SMOTE
5. Train multiple machine learning models
6. Generate comprehensive evaluation metrics
7. Create model explanations using SHAP
8. Save all results and visualizations

### Step 3: Launch the Streamlit App

```bash
streamlit run app.py
```

This will open a web interface with:
1. **Model Comparison**: View and compare the performance metrics of different models
2. **Prediction**: Enter transaction details to predict if a transaction is fraudulent

## Implemented Models

The project implements and compares four machine learning models:
1. Logistic Regression (baseline)
2. Random Forest
3. XGBoost
4. LightGBM

## Evaluation Metrics

Models are evaluated using:
- Accuracy
- Precision
- Recall
- F1 Score
- AUC Score
- Confusion Matrix

## Model Explainability

The project uses SHAP (SHapley Additive exPlanations) to:
- Explain model predictions
- Identify important features
- Understand feature interactions
- Generate feature importance plots

## Features

- Combines multiple datasets for comprehensive training
- Advanced feature engineering for better fraud detection
- Multiple model support with comprehensive comparison
- Interactive web interface for easy prediction
- Detailed model comparison and visualization
- Model explainability using SHAP
- Automatic handling of categorical variables
- Class balancing using SMOTE
- Model persistence for future use

## Contributing

Feel free to submit issues and enhancement requests! 