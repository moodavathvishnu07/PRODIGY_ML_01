"""
PRODIGY INFOTECH - MACHINE LEARNING INTERNSHIP
TASK-01: House Price Prediction using Linear Regression
Author: Intern
Dataset: Kaggle House Prices (Ames Housing)

Problem Statement:
Implement a linear regression model to predict the prices of houses 
based on their square footage and the number of bedrooms and bathrooms.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

plt.style.use('default')

def run_pipeline():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "data", "train.csv")
    plots_dir = os.path.join(base_dir, "plots")
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    print("=" * 70)
    print("      PRODIGY INFOTECH - ML INTERNSHIP: TASK 01")
    print("      House Price Prediction using Linear Regression")
    print("=" * 70)

    # 1. LOAD DATASET
    if not os.path.exists(data_path):
        from dataset import load_or_create_dataset
        df = load_or_create_dataset()
    else:
        df = pd.read_csv(data_path)
    
    print(f"\n[1] Dataset Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # 2. EXPLORATORY DATA ANALYSIS (EDA)
    key_features = ["GrLivArea", "BedroomAbvGr", "FullBath", "HalfBath", "SalePrice"]
    print("\n[2] Summary Statistics of Key Features:")
    print(df[key_features].describe().round(2).to_string())

    # 3. FEATURE SELECTION & DEFINITION
    X = df[["GrLivArea", "BedroomAbvGr", "FullBath", "HalfBath"]]
    y = df["SalePrice"]

    # 4. TRAIN-TEST SPLIT (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"\n[3] Data Split:")
    print(f"    - Training Set: {X_train.shape[0]} samples")
    print(f"    - Testing Set : {X_test.shape[0]} samples")

    # 5. MODEL TRAINING
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # 6. MODEL EVALUATION
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)
    rmse_test = np.sqrt(mse_test)
    mape_test = np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100

    print("\n" + "=" * 70)
    print("                    MODEL PERFORMANCE METRICS")
    print("=" * 70)
    print(f"  Training R^2 Score           : {r2_train:.4f} ({r2_train*100:.2f}%)")
    print(f"  Testing R^2 Score            : {r2_test:.4f} ({r2_test*100:.2f}%)")
    print(f"  Mean Absolute Error (MAE)   : ${mae_test:,.2f}")
    print(f"  Mean Squared Error (MSE)    : {mse_test:,.2f}")
    print(f"  Root Mean Squared Error(RMSE): ${rmse_test:,.2f}")
    print(f"  Mean Absolute % Error (MAPE): {mape_test:.2f}%")
    print("=" * 70)

    # 7. REGRESSION EQUATION & COEFFICIENTS
    print("\n[4] Linear Regression Equation:")
    print(f"  SalePrice = ${model.intercept_:,.2f}")
    for feature, coef in zip(X.columns, model.coef_):
        sign = "+" if coef >= 0 else "-"
        print(f"              {sign} ({abs(coef):,.2f} * {feature})")

    # 8. GENERATE VISUALIZATIONS
    print("\n[5] Generating Visualizations...")

    # Plot 1: Actual vs Predicted Prices
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test, y_test_pred, color="#1f77b4", alpha=0.6, edgecolors='black', s=45, label="Test Predictions")
    min_val = min(y_test.min(), y_test_pred.min())
    max_val = max(y_test.max(), y_test_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2.5, label=f"Perfect Fit Line (R^2 = {r2_test:.3f})")
    ax.set_title("Actual vs Predicted House Prices (Task-01)", fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel("Actual Sale Price ($)", fontsize=12)
    ax.set_ylabel("Predicted Sale Price ($)", fontsize=12)
    ax.legend(frameon=True, fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    p1 = os.path.join(plots_dir, "1_actual_vs_predicted.png")
    fig.savefig(p1, dpi=100)
    plt.close(fig)
    print(f"  [+] Saved: {p1}")

    # Plot 2: Residuals Distribution
    residuals = y_test - y_test_pred
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(residuals, bins=30, color="#2ca02c", edgecolor='black', alpha=0.7, density=True)
    # Add normal distribution curve
    mu, std = residuals.mean(), residuals.std()
    x_range = np.linspace(residuals.min(), residuals.max(), 100)
    p = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_range - mu) / std) ** 2)
    ax.plot(x_range, p, 'r-', lw=2.5, label=f"Normal Curve (\u03bc={mu:,.0f}, \u03c3={std:,.0f})")
    ax.axvline(x=0, color='black', linestyle='--', linewidth=2, label="Zero Error Line")
    ax.set_title("Residuals (Error) Distribution", fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel("Residuals ($)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.legend(frameon=True, fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    p2 = os.path.join(plots_dir, "2_residuals_distribution.png")
    fig.savefig(p2, dpi=100)
    plt.close(fig)
    print(f"  [+] Saved: {p2}")

    # Plot 3: Feature Correlation Heatmap
    corr = df[key_features].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.matshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    fig.colorbar(cax, label="Correlation Coefficient")
    ax.set_xticks(range(len(key_features)))
    ax.set_yticks(range(len(key_features)))
    ax.set_xticklabels(key_features, rotation=45, ha="left", fontsize=10, fontweight="bold")
    ax.set_yticklabels(key_features, fontsize=10, fontweight="bold")
    for i in range(len(key_features)):
        for j in range(len(key_features)):
            val = corr.iloc[i, j]
            color = "white" if abs(val) > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontweight="bold")
    ax.set_title("Feature Correlation Matrix with SalePrice", fontsize=13, fontweight='bold', pad=20)
    plt.tight_layout()
    p3 = os.path.join(plots_dir, "3_correlation_heatmap.png")
    fig.savefig(p3, dpi=100)
    plt.close(fig)
    print(f"  [+] Saved: {p3}")

    # Plot 4: Feature Coefficients
    fig, ax = plt.subplots(figsize=(9, 5))
    feature_labels = ["GrLivArea (SqFt)", "BedroomAbvGr (Beds)", "FullBath (Full Baths)", "HalfBath (Half Baths)"]
    coefs = model.coef_
    bars = ax.barh(feature_labels, coefs, color='#3498db', edgecolor='black', height=0.5)
    ax.set_title("Linear Regression Coefficients (Feature Impact on Price)", fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel("Dollar Value per Unit Increase ($)", fontsize=12)
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 500, bar.get_y() + bar.get_height()/2, f"${width:,.2f}", va='center', ha='left', fontsize=10, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5, axis='x')
    plt.tight_layout()
    p4 = os.path.join(plots_dir, "4_feature_coefficients.png")
    fig.savefig(p4, dpi=100)
    plt.close(fig)
    print(f"  [+] Saved: {p4}")

    # Plot 5: Square Footage vs Price
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df["GrLivArea"], df["SalePrice"], alpha=0.45, color='#2b5c8f', edgecolors='none', s=35, label="House Data")
    # Linear trend
    m_val, b_val = np.polyfit(df["GrLivArea"], df["SalePrice"], 1)
    sq_range = np.linspace(df["GrLivArea"].min(), df["GrLivArea"].max(), 100)
    ax.plot(sq_range, m_val * sq_range + b_val, color='#e74c3c', lw=2.5, label=f"Trendline (Slope: ${m_val:.1f}/sqft)")
    ax.set_title("House Price vs Living Area Square Footage", fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel("Ground Living Area (Sq Ft)", fontsize=12)
    ax.set_ylabel("Sale Price ($)", fontsize=12)
    ax.legend(frameon=True, fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    p5 = os.path.join(plots_dir, "5_price_vs_sqft_scatter.png")
    fig.savefig(p5, dpi=100)
    plt.close(fig)
    print(f"  [+] Saved: {p5}")

    # 9. SAVE TRAINED MODEL
    model_save_path = os.path.join(models_dir, "linear_regression_model.joblib")
    joblib.dump(model, model_save_path)
    print(f"\n[6] Model saved to: {model_save_path}")

    # 10. SAMPLE PREDICTIONS
    print("\n[7] Sample Predictions for Demonstration:")
    sample_houses = pd.DataFrame([
        {"GrLivArea": 1200, "BedroomAbvGr": 2, "FullBath": 1, "HalfBath": 0},
        {"GrLivArea": 1800, "BedroomAbvGr": 3, "FullBath": 2, "HalfBath": 1},
        {"GrLivArea": 2500, "BedroomAbvGr": 4, "FullBath": 3, "HalfBath": 1},
        {"GrLivArea": 3200, "BedroomAbvGr": 5, "FullBath": 3, "HalfBath": 2},
    ])
    sample_preds = model.predict(sample_houses)
    for i, (_, row) in enumerate(sample_houses.iterrows()):
        print(f"  - House {i+1}: {int(row['GrLivArea']):,} sq ft | {int(row['BedroomAbvGr'])} Beds | {int(row['FullBath'])} Full Bath | {int(row['HalfBath'])} Half Bath => Predicted Price: ${sample_preds[i]:,.2f}")

    print("\n[+] Pipeline execution finished successfully!\n")
    return model

if __name__ == "__main__":
    run_pipeline()
