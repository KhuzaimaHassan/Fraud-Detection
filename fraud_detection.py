import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    classification_report,
)
from imblearn.over_sampling import SMOTE
import joblib
import os
import warnings

warnings.filterwarnings("ignore")

# Create necessary directories if they don't exist
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)


def engineer_features(df):
    """Feature engineering for fraud detection"""
    df["transaction_diff"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
    df["dest_diff"] = df["newbalanceDest"] - df["oldbalanceDest"]
    df["is_zero_balance"] = (
        (df["oldbalanceOrg"] == 0) & (df["newbalanceOrig"] == 0)
    ).astype(int)
    df["amount_balance_ratio"] = df["amount"] / (df["oldbalanceOrg"] + 1)
    df["balance_change_ratio"] = (df["newbalanceOrig"] - df["oldbalanceOrg"]) / (
        df["oldbalanceOrg"] + 1
    )
    return df


def preprocess_data(file_path):
    """Load and preprocess data using existing label encoder"""
    df = pd.read_csv(file_path)

    # Drop unnecessary columns
    if "nameOrig" in df.columns:
        df = df.drop(["nameOrig", "nameDest"], axis=1)

    # Feature engineering
    df = engineer_features(df)

    # Load existing label encoder
    le = joblib.load("models/label_encoder.pkl")
    df["type"] = le.transform(df["type"])

    # Prepare features and target
    if "isFlaggedFraud" in df.columns:
        X = df.drop(["isFraud", "isFlaggedFraud"], axis=1)
        y = df["isFraud"]
    else:
        X = df.drop(["isFraud"], axis=1)
        y = df["isFraud"]

    return X, y


def evaluate_existing_models(X_test_scaled, y_test):
    """Evaluate all existing models without retraining"""
    model_files = {
        "Logistic Regression": "logistic_regression.pkl",
        "Random Forest": "random_forest.pkl",
        "XGBoost": "xgboost.pkl",
        "LightGBM": "lightgbm.pkl",
        "SVM": "svm.pkl",  # This will fail gracefully if not found
    }

    results = {}

    for name, file in model_files.items():
        model_path = f"models/{file}"
        if not os.path.exists(model_path):
            print(f"\nModel not found: {name} ({model_path})")
            continue

        print(f"\nEvaluating existing {name} model...")
        model = joblib.load(model_path)

        # Predictions
        try:
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        except Exception as e:
            print(f"Error making predictions with {name}: {str(e)}")
            continue

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        cm = confusion_matrix(y_test, y_pred)

        # Save metrics
        results[name] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc": auc,
            "confusion_matrix": cm,
        }

        # Plot and save confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix - {name}")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plot_path = f'results/confusion_matrix_{name.replace(" ", "_").lower()}.png'
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved confusion matrix to {plot_path}")

        # Print classification report
        print(f"\nClassification Report for {name}:")
        print(classification_report(y_test, y_pred))

    return results


def main():
    """Main execution function"""
    print("Loading and preprocessing data...")
    try:
        X_full, y_full = preprocess_data("Fraud.csv")
        X_sampled, y_sampled = preprocess_data("FraudData_sampled.csv")
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return

    # Combine datasets
    X_combined = pd.concat([X_full, X_sampled], ignore_index=True)
    y_combined = pd.concat([y_full, y_sampled], ignore_index=True)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, y_combined, test_size=0.2, random_state=42, stratify=y_combined
    )

    # Load existing scaler
    scaler = joblib.load("models/scaler.pkl")
    X_test_scaled = scaler.transform(X_test)

    # Evaluate existing models
    print("\nEvaluating existing models...")
    results = evaluate_existing_models(X_test_scaled, y_test)

    print("\nProcess completed successfully!")
    print("\nModel Evaluation Summary:")
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print(f"AUC: {metrics['auc']:.4f}")


if __name__ == "__main__":
    main()
