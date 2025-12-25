
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from data_loader import download_data

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
IMG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'notebooks', 'images')

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
if not os.path.exists(IMG_DIR):
    os.makedirs(IMG_DIR)

def train_models():
    print("Loading data...")
    df = download_data()
    if df is None:
        print("Could not load data. Exiting.")
        return

    # Renaming columns for clarity if needed, but assuming standard UCi structure
    # X1..X8 are features, Y1 is Heating Load, Y2 is Cooling Load
    feature_cols = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8']
    target_cols = ['Y1', 'Y2']
    
    # Ensure columns exist, else use indices or rely on standard load
    if not all(col in df.columns for col in feature_cols + target_cols):
        # Taking a guess if column names differ (e.g. from Excel)
        print("Column names might differ. Using first 8 as features and last 2 as targets.")
        X = df.iloc[:, :8]
        y = df.iloc[:, 8:10]
        X.columns = feature_cols
        y.columns = target_cols
    else:
        X = df[feature_cols]
        y = df[target_cols]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Dictionary to store models
    models = {}
    
    print("\nTraining Models...")
    
    # We will train separate models for Y1 (Heating) and Y2 (Cooling) or multi-output.
    # RF and XGBoost support multi-output naturally or wrapper.
    # Checking performance for each target is better.
    
    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    y_pred_lr = lr.predict(X_test_scaled)
    # R2
    r2_lr = r2_score(y_test, y_pred_lr)
    print(f"Linear Regression R2: {r2_lr:.4f}")
    models['LinearRegression'] = lr

    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train_scaled, y_train)
    y_pred_rf = rf.predict(X_test_scaled)
    r2_rf = r2_score(y_test, y_pred_rf)
    print(f"Random Forest R2: {r2_rf:.4f}")
    models['RandomForest'] = rf

    # XGBoost
    # XGBoost Regressor for multi-output is supported via MultiOutputRegressor usually, 
    # but the native API or sklearn wrapper might need specific handling. 
    # XGBRegressor supports single target. For multi-target, we can wrap or train two models.
    # Let's use MultiOutputRegressor or just stick to RF which handles it easily. 
    # Or train XGB for each. Let's start with RF as primary for simplicity in this demo script.
    
    # Save best model (RF usually performs well here)
    best_model = rf
    
    # --- Comprehensive Visualization ---
    print("\nGenerating Comprehensive Evaluation Plots...")
    plt.figure(figsize=(18, 12))
    
    # 1. Linear Regression: Actual vs Predicted (Heating)
    plt.subplot(2, 3, 1)
    plt.scatter(y_test.iloc[:, 0], y_pred_lr[:, 0], alpha=0.5, color='blue', label='Predictions')
    plt.plot([y_test.iloc[:, 0].min(), y_test.iloc[:, 0].max()], [y_test.iloc[:, 0].min(), y_test.iloc[:, 0].max()], 'r--', label='Ideal')
    plt.title(f"Linear Regression: Heating Load\nR2: {r2_score(y_test.iloc[:, 0], y_pred_lr[:, 0]):.4f}")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.legend()

    # 2. Linear Regression: Actual vs Predicted (Cooling)
    plt.subplot(2, 3, 2)
    plt.scatter(y_test.iloc[:, 1], y_pred_lr[:, 1], alpha=0.5, color='cyan', label='Predictions')
    plt.plot([y_test.iloc[:, 1].min(), y_test.iloc[:, 1].max()], [y_test.iloc[:, 1].min(), y_test.iloc[:, 1].max()], 'r--', label='Ideal')
    plt.title(f"Linear Regression: Cooling Load\nR2: {r2_score(y_test.iloc[:, 1], y_pred_lr[:, 1]):.4f}")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.legend()

    # 3. Random Forest: Actual vs Predicted (Heating)
    plt.subplot(2, 3, 4)
    plt.scatter(y_test.iloc[:, 0], y_pred_rf[:, 0], alpha=0.5, color='green', label='Predictions')
    plt.plot([y_test.iloc[:, 0].min(), y_test.iloc[:, 0].max()], [y_test.iloc[:, 0].min(), y_test.iloc[:, 0].max()], 'r--', label='Ideal')
    plt.title(f"Random Forest: Heating Load\nR2: {r2_score(y_test.iloc[:, 0], y_pred_rf[:, 0]):.4f}")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.legend()

    # 4. Random Forest: Actual vs Predicted (Cooling)
    plt.subplot(2, 3, 5)
    plt.scatter(y_test.iloc[:, 1], y_pred_rf[:, 1], alpha=0.5, color='lime', label='Predictions')
    plt.plot([y_test.iloc[:, 1].min(), y_test.iloc[:, 1].max()], [y_test.iloc[:, 1].min(), y_test.iloc[:, 1].max()], 'r--', label='Ideal')
    plt.title(f"Random Forest: Cooling Load\nR2: {r2_score(y_test.iloc[:, 1], y_pred_rf[:, 1]):.4f}")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.legend()

    # 5. Residuals Distribution (Random Forest - Heating)
    plt.subplot(2, 3, 3)
    residuals_heating = y_test.iloc[:, 0] - y_pred_rf[:, 0]
    sns.histplot(residuals_heating, kde=True, color='purple')
    plt.title(f"RF Heating Residuals\nRMSE: {np.sqrt(mean_squared_error(y_test.iloc[:, 0], y_pred_rf[:, 0])):.4f}")
    plt.xlabel("Residual")

    # 6. Residuals Distribution (Random Forest - Cooling)
    plt.subplot(2, 3, 6)
    residuals_cooling = y_test.iloc[:, 1] - y_pred_rf[:, 1]
    sns.histplot(residuals_cooling, kde=True, color='magenta')
    plt.title(f"RF Cooling Residuals\nRMSE: {np.sqrt(mean_squared_error(y_test.iloc[:, 1], y_pred_rf[:, 1])):.4f}")
    plt.xlabel("Residual")

    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, 'model_evaluation_dashboard.png'))
    plt.close()
    print(f"Comprehensive evaluation plots saved to {os.path.join(IMG_DIR, 'model_evaluation_dashboard.png')}")

    # Save artifacts
    print(f"\nSaving models to {MODEL_DIR}...")
    with open(os.path.join(MODEL_DIR, 'model_rf.pkl'), 'wb') as f:
        pickle.dump(best_model, f)
    
    with open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
        
    print("Training complete.")

if __name__ == "__main__":
    train_models()
