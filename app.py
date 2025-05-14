import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from PIL import Image

# Set page config (must be first Streamlit command)
st.set_page_config(
    page_title="Fraud Detection System", layout="wide", initial_sidebar_state="expanded"
)


# Function to load models and data with error handling
@st.cache_resource
def load_components():
    """Load all required components with progress tracking"""
    components = {}
    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()

    # 1. Load label encoder
    status_text.text("Loading label encoder...")
    try:
        components["le"] = joblib.load("models/label_encoder.pkl")
        progress_bar.progress(0.25)  # Fixed: using float between 0-1
    except Exception as e:
        st.error(f"Failed to load label encoder: {str(e)}")
        return None

    # 2. Load scaler
    status_text.text("Loading scaler...")
    try:
        components["scaler"] = joblib.load("models/scaler.pkl")
        progress_bar.progress(0.5)  # Fixed: using float between 0-1
    except Exception as e:
        st.error(f"Failed to load scaler: {str(e)}")
        return None

    # 3. Load models
    components["models"] = {}
    model_files = {
        "Logistic Regression": "logistic_regression.pkl",
        "Random Forest": "random_forest.pkl",
        "XGBoost": "xgboost.pkl",
        "LightGBM": "lightgbm.pkl",
    }

    # Calculate progress increments
    base_progress = 0.5
    progress_increment = 0.5 / len(model_files)

    for i, (name, file) in enumerate(model_files.items()):
        status_text.text(f"Loading {name}...")
        try:
            components["models"][name] = joblib.load(f"models/{file}")
            progress_bar.progress(base_progress + ((i + 1) * progress_increment))
        except Exception as e:
            st.error(f"Failed to load {name}: {str(e)}")
            continue

    progress_bar.progress(1.0)
    status_text.text("Loading complete!")
    return components


@st.cache_data
def load_results():
    """Load model comparison results"""
    try:
        results = pd.read_csv("results/model_comparison.csv")
        # Filter out SVM if present
        return results[~results["Model"].str.contains("SVM")]
    except Exception as e:
        st.error(f"Failed to load results: {str(e)}")
        return None


def display_model_comparison(results_df):
    """Display model comparison page"""
    st.header("Model Performance Comparison")

    if results_df is None:
        st.warning("No model results available")
        return

    # Metrics table
    st.subheader("Evaluation Metrics")
    st.dataframe(
        results_df.style.format(
            {
                "Accuracy": "{:.4f}",
                "Precision": "{:.4f}",
                "Recall": "{:.4f}",
                "F1": "{:.4f}",
                "AUC": "{:.4f}",
            }
        )
    )

    # Visual comparison
    st.subheader("Performance Visualization")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    metrics = ["Accuracy", "Precision", "Recall", "F1"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for i, (metric, color) in enumerate(zip(metrics, colors)):
        ax = axes[i // 2, i % 2]
        sns.barplot(x="Model", y=metric, data=results_df, ax=ax, color=color)
        ax.set_title(f"{metric} Comparison", fontsize=14)
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    st.pyplot(fig)

    # Confusion matrices
    st.subheader("Confusion Matrices")
    cols = st.columns(2)
    for i, model_name in enumerate(results_df["Model"]):
        col = cols[i % 2]
        try:
            img_path = (
                f'results/confusion_matrix_{model_name.replace(" ", "_").lower()}.png'
            )
            img = Image.open(img_path)
            col.image(img, caption=f"{model_name}", use_column_width=True)
        except FileNotFoundError:
            col.warning(f"Image not found for {model_name}")


def make_prediction(components):
    """Make predictions using loaded models"""
    st.header("Transaction Fraud Prediction")
    st.write("Enter the transaction details below to check for potential fraud.")

    # Input fields organized in columns
    col1, col2 = st.columns(2)

    with col1:
        step = st.number_input(
            "Hour of Transaction (1-744)", min_value=1, max_value=744, value=100
        )
        transaction_type = st.selectbox(
            "Transaction Type", ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
        )
        amount = st.number_input("Amount ($)", min_value=0.0, value=1000.0, step=0.01)
        oldbalanceOrg = st.number_input(
            "Origin Account Old Balance", min_value=0.0, value=5000.0, step=0.01
        )

    with col2:
        newbalanceOrig = st.number_input(
            "Origin Account New Balance", min_value=0.0, value=4000.0, step=0.01
        )
        oldbalanceDest = st.number_input(
            "Destination Account Old Balance", min_value=0.0, value=1000.0, step=0.01
        )
        newbalanceDest = st.number_input(
            "Destination Account New Balance", min_value=0.0, value=2000.0, step=0.01
        )

    # Feature engineering (must match training)
    transaction_diff = oldbalanceOrg - newbalanceOrig
    dest_diff = newbalanceDest - oldbalanceDest
    is_zero_balance = 1 if (oldbalanceOrg == 0 and newbalanceOrig == 0) else 0
    amount_balance_ratio = amount / (oldbalanceOrg + 1e-6)  # Avoid division by zero
    balance_change_ratio = (newbalanceOrig - oldbalanceOrg) / (oldbalanceOrg + 1e-6)

    # Select model
    model_name = st.selectbox("Select Model", list(components["models"].keys()))

    if st.button("Predict Fraud Risk", type="primary"):
        try:
            # Encode transaction type
            type_encoded = components["le"].transform([transaction_type])[0]

            # Create input DataFrame with EXACTLY the same features as training
            input_data = {
                "step": step,
                "type": type_encoded,
                "amount": amount,
                "oldbalanceOrg": oldbalanceOrg,
                "newbalanceOrig": newbalanceOrig,
                "oldbalanceDest": oldbalanceDest,
                "newbalanceDest": newbalanceDest,
                "transaction_diff": transaction_diff,
                "dest_diff": dest_diff,
                "is_zero_balance": is_zero_balance,
                "amount_balance_ratio": amount_balance_ratio,
                "balance_change_ratio": balance_change_ratio,
            }

            # Convert to DataFrame with correct column order
            input_df = pd.DataFrame(
                [input_data],
                columns=[
                    "step",
                    "type",
                    "amount",
                    "oldbalanceOrg",
                    "newbalanceOrig",
                    "oldbalanceDest",
                    "newbalanceDest",
                    "transaction_diff",
                    "dest_diff",
                    "is_zero_balance",
                    "amount_balance_ratio",
                    "balance_change_ratio",
                ],
            )

            # Scale features
            input_scaled = components["scaler"].transform(input_df)

            # Make prediction
            model = components["models"][model_name]
            prediction = model.predict(input_scaled)[0]
            proba = model.predict_proba(input_scaled)[0][1]

            # Display results
            st.subheader("Prediction Result")
            if prediction == 1:
                st.error(f"🚨 **Fraud Detected** (Probability: {proba:.2%})")
                st.warning(
                    "This transaction has been flagged as potentially fraudulent."
                )
            else:
                st.success(
                    f"✅ **Legitimate Transaction** (Fraud Probability: {proba:.2%})"
                )
                st.info("This transaction appears to be legitimate.")

            # Show feature importance if available
            if hasattr(model, "feature_importances_"):
                st.subheader("Key Influencing Factors")
                feature_importance = pd.DataFrame(
                    {
                        "Feature": input_df.columns,
                        "Importance": model.feature_importances_,
                    }
                ).sort_values("Importance", ascending=False)

                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x="Importance", y="Feature", data=feature_importance, ax=ax)
                ax.set_title("Feature Importance")
                st.pyplot(fig)

        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")


def main():
    """Main application flow"""
    st.title("Financial Fraud Detection System")

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Model Comparison", "Fraud Prediction"])

    # Load components
    components = load_components()
    if components is None:
        st.error("Critical components failed to load. Please check the error messages.")
        return

    # Load results
    results_df = load_results()

    # Page routing
    if page == "Model Comparison":
        display_model_comparison(results_df)
    else:
        make_prediction(components)

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info(
        "Fraud Detection System v1.0\n\n"
        "This application uses machine learning models to detect potentially fraudulent financial transactions."
    )


if __name__ == "__main__":
    main()
