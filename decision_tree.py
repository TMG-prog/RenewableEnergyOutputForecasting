"""
Member 5: Decision Tree Regressor

This script trains two Decision Tree Regression models:
1. Wind Power Proxy
2. Solar Power Proxy

It performs hyperparameter tuning using GridSearchCV,
evaluates the models using MAE, RMSE and R²,
and saves the models, predictions, plots, and metrics.
"""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# ----------------------------------------------------
# Configuration
# ----------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent

DATA_DIR = PROJECT_DIR / "Project" / "data" / "train"

MODELS_DIR = PROJECT_DIR / "models" / "decision_tree"

REPORTS_DIR = PROJECT_DIR / "reports" / "decision_tree"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------
# Load Dataset
# ----------------------------------------------------

print("Loading dataset...")

X_train = pd.read_csv(DATA_DIR / "X_train.csv")
X_test = pd.read_csv(DATA_DIR / "X_test.csv")

y_train = pd.read_csv(DATA_DIR / "y_train.csv")
y_test = pd.read_csv(DATA_DIR / "y_test.csv")

print(f"Training rows : {len(X_train)}")
print(f"Testing rows  : {len(X_test)}")

# ----------------------------------------------------
# Keep only numeric columns
# ----------------------------------------------------

numeric_columns = X_train.select_dtypes(
    include=["number", "bool"]
).columns

X_train = X_train[numeric_columns]
X_test = X_test[numeric_columns]

print(f"Using {len(numeric_columns)} numeric features.")

# ----------------------------------------------------
# Helper: save actual-vs-predicted and residual plots
# ----------------------------------------------------

def save_plots(y_true, y_pred, name):
    """Save an actual-vs-predicted plot and a residual plot for one target."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Actual vs Predicted
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.3, s=14)
    plt.plot(
        [y_true.min(), y_true.max()],
        [y_true.min(), y_true.max()],
        "r--",
        label="Perfect prediction",
    )
    plt.xlabel("Actual value")
    plt.ylabel("Predicted value")
    plt.title(f"Actual vs Predicted — {name}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / f"{name.lower()}_actual_vs_predicted.png", dpi=300)
    plt.close()

    # Residual plot
    residuals = y_true - y_pred
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, alpha=0.3, s=14)
    plt.axhline(y=0, color="r", linestyle="--")
    plt.xlabel("Predicted value")
    plt.ylabel("Residual")
    plt.title(f"Residual Plot — {name}")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / f"{name.lower()}_residual_plot.png", dpi=300)
    plt.close()


def save_feature_importance(model, feature_names, name):
    """Save a feature importance table (CSV) and a bar chart (PNG)."""
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_,
    }).sort_values(by="importance", ascending=False)

    importance_df.to_csv(
        REPORTS_DIR / f"{name.lower()}_feature_importance.csv", index=False
    )

    top_features = importance_df.head(10)
    plt.figure(figsize=(9, 6))
    plt.barh(top_features["feature"], top_features["importance"])
    plt.xlabel("Importance")
    plt.title(f"Top 10 Feature Importances — {name}")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / f"{name.lower()}_feature_importance.png", dpi=300)
    plt.close()


# ----------------------------------------------------
# Wind Power Decision Tree
# ----------------------------------------------------

print("\nTraining Wind Power Model...")

# Hyperparameter values to test
param_grid = {
    "max_depth": [5, 10, None],
    "min_samples_leaf": [1, 5, 10, 20, 30, 50]
}

# Create the Decision Tree
wind_tree = DecisionTreeRegressor(
    random_state=42
)

# Perform hyperparameter tuning
wind_grid = GridSearchCV(
    estimator=wind_tree,
    param_grid=param_grid,
    cv=5,
    scoring="neg_mean_absolute_error",
    n_jobs=-1
)

# Train the model
wind_grid.fit(
    X_train,
    y_train["wind_power_proxy"]
)

# Best model after tuning
wind_model = wind_grid.best_estimator_
wind_best_params = wind_grid.best_params_

print("Best Parameters:")
print(wind_best_params)

# Predict Wind Power

wind_predictions = wind_model.predict(X_test)

# Renewable energy cannot be negative
wind_predictions = np.clip(
    wind_predictions,
    a_min=0,
    a_max=None
)

# Evaluation Metrics

wind_mae = mean_absolute_error(
    y_test["wind_power_proxy"],
    wind_predictions
)

wind_rmse = np.sqrt(
    mean_squared_error(
        y_test["wind_power_proxy"],
        wind_predictions
    )
)

wind_r2 = r2_score(
    y_test["wind_power_proxy"],
    wind_predictions
)

print("\nWind Model Results")

print(f"MAE  : {wind_mae:.4f}")
print(f"RMSE : {wind_rmse:.4f}")
print(f"R²   : {wind_r2:.4f}")

# Save Wind Model

joblib.dump(
    wind_model,
    MODELS_DIR / "decision_tree_wind.pkl"
)

# Save wind plots and feature importance

save_plots(y_test["wind_power_proxy"], wind_predictions, "Wind")
save_feature_importance(wind_model, X_train.columns, "Wind")

# ----------------------------------------------------
# Solar Power Decision Tree
# ----------------------------------------------------

print("\nTraining Solar Power Model...")

# Create the Decision Tree
solar_tree = DecisionTreeRegressor(
    random_state=42
)

# Hyperparameter tuning
solar_grid = GridSearchCV(
    estimator=solar_tree,
    param_grid=param_grid,
    cv=5,
    scoring="neg_mean_absolute_error",
    n_jobs=-1
)

# Train the model
solar_grid.fit(
    X_train,
    y_train["solar_power_proxy"]
)

# Best model
solar_model = solar_grid.best_estimator_
solar_best_params = solar_grid.best_params_

print("Best Parameters:")
print(solar_best_params)

# Predict Solar Power

solar_predictions = solar_model.predict(X_test)

# Renewable energy cannot be negative
solar_predictions = np.clip(
    solar_predictions,
    a_min=0,
    a_max=None
)

# Evaluation Metrics

solar_mae = mean_absolute_error(
    y_test["solar_power_proxy"],
    solar_predictions
)

solar_rmse = np.sqrt(
    mean_squared_error(
        y_test["solar_power_proxy"],
        solar_predictions
    )
)

solar_r2 = r2_score(
    y_test["solar_power_proxy"],
    solar_predictions
)

print("\nSolar Model Results")

print(f"MAE  : {solar_mae:.4f}")
print(f"RMSE : {solar_rmse:.4f}")
print(f"R²   : {solar_r2:.4f}")

# Save Solar Model

joblib.dump(
    solar_model,
    MODELS_DIR / "decision_tree_solar.pkl"
)

# Save solar plots and feature importance

save_plots(y_test["solar_power_proxy"], solar_predictions, "Solar")
save_feature_importance(solar_model, X_train.columns, "Solar")

# ----------------------------------------------------
# Save Predictions
# ----------------------------------------------------

predictions = pd.DataFrame({
    "Actual_Wind": y_test["wind_power_proxy"],
    "Predicted_Wind": wind_predictions,
    "Actual_Solar": y_test["solar_power_proxy"],
    "Predicted_Solar": solar_predictions,
})

predictions.to_csv(
    REPORTS_DIR / "decision_tree_predictions.csv",
    index=False
)

# ----------------------------------------------------
# Save Metrics as JSON (for Member 8's model comparison)
# ----------------------------------------------------

metrics = {
    "model_type": "Decision Tree Regressor",
    "numeric_feature_count": len(numeric_columns),
    "wind_model": {
        "mae": float(wind_mae),
        "rmse": float(wind_rmse),
        "r2_score": float(wind_r2),
        "best_params": wind_best_params,
    },
    "solar_model": {
        "mae": float(solar_mae),
        "rmse": float(solar_rmse),
        "r2_score": float(solar_r2),
        "best_params": solar_best_params,
    },
}

with open(REPORTS_DIR / "decision_tree_metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=4)

print("\nDone! Models, predictions, plots, and metrics have been saved.")
print(f"Reports saved to: {REPORTS_DIR}")