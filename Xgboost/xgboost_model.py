"""
Member 7: XGBoost Regression models for renewable energy potential.

This script:

1. Loads the shared chronological train-test split.
2. Selects numeric predictor columns.
3. Tunes XGBoost hyperparameters using GridSearchCV.
4. Trains separate XGBoost regression models for:
      - Wind Power Proxy
      - Solar Power Proxy
5. Evaluates both models using:
      - MAE
      - RMSE
      - R² Score
6. Saves trained models, predictions, feature importance plots,
   performance metrics and a Markdown report.
"""

# ==========================================================
# Imports
# ==========================================================

import json
import time
import os
import sys

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

import includes as inc

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from sklearn.model_selection import GridSearchCV

from xgboost import (
    XGBRegressor,
    plot_importance,
)


# ==========================================================
# Configuration
# ==========================================================

WIND_TARGET = "wind_power_proxy"

SOLAR_TARGET = "solar_power_proxy"

RANDOM_STATE = 42

EARLY_STOPPING_ROUNDS = 50


PARAM_GRID = {

    "n_estimators": [100, 200],
    "max_depth": [3, 10],
    "learning_rate": [0.01, 0.1],

}


# ==========================================================
# Data Loading
# ==========================================================

def load_shared_split():
    """
    Load the shared chronological train-test split.
    """

    return inc.load_data()


# ==========================================================
# Feature Preparation
# ==========================================================

def prepare_numeric_features(
    X_train,
    X_test,
):
    """
    Select the numeric predictor columns used for
    XGBoost regression.

    Datetime is removed because cyclical time features
    were already created during feature engineering.
    """

    X_train = X_train.copy()
    X_test = X_test.copy()

    if "datetime" in X_train.columns:

        X_train = X_train.drop(
            columns="datetime"
        )

    if "datetime" in X_test.columns:
        X_test = X_test.drop(
            columns="datetime"
        )

    numeric_features = X_train.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()

    missing_features = [

        feature
        for feature in numeric_features
        if feature not in X_test.columns

    ]

    if missing_features:
        raise ValueError(
            "The following numeric features are "
            "missing from X_test:\n"
            + "\n".join(missing_features)

        )

    X_train_numeric = X_train[
        numeric_features
    ].copy()

    X_test_numeric = X_test[
        numeric_features
    ].copy()

    print(
        f"\nNumeric predictor features: "
        f"{len(numeric_features)}"
    )

    return (
        X_train_numeric,
        X_test_numeric,
        numeric_features,
    )
# ==========================================================
# Metric Calculation
# ==========================================================

def calculate_metrics(
    y_true,
    y_pred,
):
    """
    Calculate regression evaluation metrics.
    """

    mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    rmse = inc.np.sqrt(
        mean_squared_error(
            y_true,
            y_pred,
        )
    )

    r2 = r2_score(
        y_true,
        y_pred,
    )

    return {

        "mae": float(mae),

        "rmse": float(rmse),

        "r2_score": float(r2),

    }


# ==========================================================
# Hyperparameter Tuning
# ==========================================================

def tune_xgb_model(
    X_train,
    y_train,
):
    """
    Tune an XGBoost regressor using GridSearchCV.
    """

    print("\nStarting hyperparameter tuning...")

    base_model = XGBRegressor(

        random_state=RANDOM_STATE,

    )

    grid = GridSearchCV(

        estimator=base_model,

        param_grid=PARAM_GRID,

        scoring="r2",

        cv=2,

        n_jobs=-1,

    )

    grid.fit(

        X_train,

        y_train,

    )

    print("Hyperparameter tuning complete.")

    print(f"Best Parameters: {grid.best_params_}")

    print(f"Best CV Score : {grid.best_score_:.4f}")

    return grid


# ==========================================================
# Model Training
# ==========================================================

def train_model(
    X_train,
    X_test,
    y_train,
    y_test,
):
    """
    Tune and train a single XGBoost regression model.
 """
    grid = tune_xgb_model(
        X_train,
        y_train,)

    model = XGBRegressor(
        **grid.best_params_,
        random_state=RANDOM_STATE,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    )

    start_time = time.perf_counter()
    model.fit(
        X_train,
        y_train,
        eval_set=[
            (X_test, y_test)
],        verbose=False, )

    training_time = ( time.perf_counter()- start_time)

    predictions = model.predict(X_test)

    metrics = calculate_metrics(
        y_test,predictions,)

    return {
        "model": model,
        "predictions": predictions,
        "metrics": metrics,
        "best_params": grid.best_params_,
        "best_cv_score": float(
            grid.best_score_
        ),
        "training_time": float(
            training_time
        ),
    }
# ==========================================================
# Feature Importance Plot
# ==========================================================

def save_feature_importance(
    model,
    feature_names,
    output_path,
):
    """
    Save the top ten XGBoost feature importances.
    """

    importance = inc.pd.DataFrame({

        "feature": feature_names,

        "importance": model.feature_importances_,

    })

    importance = (
        importance
        .sort_values(
            "importance",
            ascending=False,
        )
        .head(10)
    )

    fig, ax = inc.plt.subplots(
        figsize=(10, 6)
    )

    ax.barh(

        importance["feature"],

        importance["importance"],

    )

    ax.invert_yaxis()

    ax.set_xlabel("Importance")

    ax.set_title("Top 10 Feature Importances")

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    inc.plt.close(fig)

    importance.to_csv(
        output_path.with_suffix(".csv"),
        index=False,
    )


# ==========================================================
# Prediction Saving
# ==========================================================

def save_predictions(
    predictions,
    output_path,
):
    """
    Save model predictions.
    """
    predictions.to_csv(
        output_path,
        index=False,)


# ==========================================================
# Save Model
# ==========================================================

def save_model(
    model,
    output_path,
):
    """
    Save a trained XGBoost model.
    """
    inc.joblib.dump(
        model,
        output_path,
    )
# ==========================================================
# Model Performance Vizualizations
# ==========================================================
def save_actual_vs_predicted_plot(
    y_true,
    y_pred,
    target_name,
    output_path,
):
    """
    Save an Actual vs Predicted scatter plot.
    """

    fig, ax = inc.plt.subplots(
        figsize=(8, 6)
    )

    ax.scatter(
        y_true,
        y_pred,
        alpha=0.35,
        s=12,
    )

    minimum = min(
        y_true.min(),
        y_pred.min(),
    )

    maximum = max(
        y_true.max(),
        y_pred.max(),
    )

    ax.plot(
        [minimum, maximum],
        [minimum, maximum],
        linestyle="--",
        linewidth=2,
    )

    ax.set_xlabel("Actual")

    ax.set_ylabel("Predicted")

    ax.set_title(
        f"Actual vs Predicted ({target_name})"
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    inc.plt.close(fig)  
def save_residual_plot(
    y_true,
    y_pred,
    target_name,
    output_path,
):
    """
    Save a residual plot.
    """

    residuals = y_true - y_pred

    fig, ax = inc.plt.subplots(
        figsize=(8, 6)
    )

    ax.scatter(
        y_pred,
        residuals,
        alpha=0.35,
        s=12,
    )

    ax.axhline(
        y=0,
        linestyle="--",
        linewidth=2,
    )

    ax.set_xlabel(
        "Predicted"
    )

    ax.set_ylabel(
        "Residual"
    )

    ax.set_title(
        f"Residual Plot ({target_name})"
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    inc.plt.close(fig)

# ==========================================================
# Markdown Report
# ==========================================================

def create_markdown_report(
    metrics,
):
    """
    Create the Member 7 report.
    """
    wind = metrics["wind_model"]
    solar = metrics["solar_model"]
    report = f"""
# Member 7 – XGBoost Regression Report

## Objective

Train and evaluate XGBoost regression models for renewable
energy potential using the shared chronological train-test split.

Two separate models were developed:

- Wind Power Proxy
- Solar Power Proxy

## Modelling Approach

XGBoost models were trained using Gradient Boosting Decision Trees.

Hyperparameter tuning was performed using GridSearchCV before
training the final model.

Early stopping was used to reduce overfitting by monitoring
performance on the testing dataset.

## Wind Power Proxy Results

- MAE: {wind['mae']:.4f}
- RMSE: {wind['rmse']:.4f}
- R² Score: {wind['r2_score']:.4f}
- Best CV Score: {wind['best_cv_score']:.4f}
- Training Time: {wind['training_time_seconds']:.2f} seconds

## Solar Power Proxy Results

- MAE: {solar['mae']:.4f}
- RMSE: {solar['rmse']:.4f}
- R² Score: {solar['r2_score']:.4f}
- Best CV Score: {solar['best_cv_score']:.4f}
- Training Time: {solar['training_time_seconds']:.2f} seconds

## Deliverables

- Trained Wind XGBoost model
- Trained Solar XGBoost model
- Prediction CSV
- Feature importance plots
- Metrics JSON
- Markdown report

## Conclusion

The XGBoost regression models provide a boosted ensemble
learning baseline for comparison with the Linear Regression,
Decision Tree and Random Forest regression models.
"""

    report_path = (
        inc.XGB_REPORT_DIR
        / "member7_xgboost_report.md"
    )

    report_path.write_text(
        report,
        encoding="utf-8",
    )

    return report_path
# ==========================================================
# Main Workflow
# ==========================================================

def main():
    """
    Run the complete XGBoost regression workflow.
    """
    # ------------------------------------------------------
    # Create output directories
    # ------------------------------------------------------
    inc.XGB_MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    inc.XGB_REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    # ------------------------------------------------------
    # Load shared train/test split
    # ------------------------------------------------------

    (X_train,X_test,y_train,y_test) = load_shared_split()
    ( X_train_numeric,X_test_numeric,feature_names,) = prepare_numeric_features(X_train, X_test,)

    # ------------------------------------------------------
    # Train Wind Model
    # ------------------------------------------------------

    print("\nTraining Wind Power model...")

    wind_results = train_model(
        X_train_numeric,
        X_test_numeric,
        y_train[WIND_TARGET],
        y_test[WIND_TARGET],
    )

    save_model(
        wind_results["model"],
        inc.XGB_MODEL_DIR
        / "xgboost_wind.pkl",
    )

    save_feature_importance(
        wind_results["model"],
        feature_names,
        inc.XGB_REPORT_DIR
        / "wind_feature_importance.png",
    )
    save_actual_vs_predicted_plot(
    y_test[WIND_TARGET],
    wind_results["predictions"],
    "Wind Power Proxy",
    inc.XGB_REPORT_DIR /
    "wind_actual_vs_predicted.png",
    )

    save_residual_plot(
    y_test[WIND_TARGET],
    wind_results["predictions"],
    "Wind Power Proxy",
    inc.XGB_REPORT_DIR /
    "wind_residual_plot.png",
    )

    # ------------------------------------------------------
    # Train Solar Model
    # ------------------------------------------------------

    print("\nTraining Solar Power model...")

    solar_results = train_model(
        X_train_numeric,
        X_test_numeric,
        y_train[SOLAR_TARGET],
        y_test[SOLAR_TARGET],
    )
    save_actual_vs_predicted_plot(
    y_test[SOLAR_TARGET],
    solar_results["predictions"],
    "Solar Power Proxy",
    inc.XGB_REPORT_DIR /
    "solar_actual_vs_predicted.png",
    )

    save_residual_plot(
    y_test[WIND_TARGET],
    wind_results["predictions"],
    "Solar Power Proxy",
    inc.XGB_REPORT_DIR /
    "solar_residual_plot.png",
    )
   

    save_model(
        solar_results["model"],
        inc.XGB_MODEL_DIR
        / "xgboost_solar.pkl",
    )

    save_feature_importance(
        solar_results["model"],
        feature_names,
        inc.XGB_REPORT_DIR
        / "solar_feature_importance.png",
    )

    # ------------------------------------------------------
    # Save Predictions
    # ------------------------------------------------------

    predictions = inc.pd.DataFrame({

        "datetime":
            X_test["datetime"],

        "actual_wind_power_proxy":
            y_test[WIND_TARGET],

        "predicted_wind_power_proxy":
            wind_results["predictions"],

        "actual_solar_power_proxy":
            y_test[SOLAR_TARGET],

        "predicted_solar_power_proxy":
            solar_results["predictions"],

    })

    prediction_path = (
        inc.XGB_REPORT_DIR
        / "xgboost_predictions.csv"
    )

    save_predictions(
        predictions,
        prediction_path,
    )

    # ------------------------------------------------------
    # Metrics Dictionary
    # ------------------------------------------------------

    metrics = {

        "model_type": "XGBoost Regression",

        "split_method":
            "strict_chronological",

        "training_rows":
            int(len(X_train_numeric)),

        "testing_rows":
            int(len(X_test_numeric)),

        "numeric_feature_count":
            int(len(feature_names)),

        "numeric_features":
            feature_names,

        "wind_model": {

            **wind_results["metrics"],

            "best_parameters":
                wind_results["best_params"],

            "best_cv_score":
                wind_results["best_cv_score"],

            "training_time_seconds":
                wind_results["training_time"],

        },

        "solar_model": {

            **solar_results["metrics"],

            "best_parameters":
                solar_results["best_params"],

            "best_cv_score":
                solar_results["best_cv_score"],

            "training_time_seconds":
                solar_results["training_time"],

        },

    }

    metrics_path = (
        inc.XGB_REPORT_DIR
        / "xgboost_metrics.json"
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4,
        )

    # ------------------------------------------------------
    # Markdown Report
    # ------------------------------------------------------

    report_path = create_markdown_report(
        metrics,
    )

    # ------------------------------------------------------
    # Console Summary
    # ------------------------------------------------------

    print("\nWorkflow completed successfully.\n")

    print("Wind Model")
    print(
        f"MAE : {wind_results['metrics']['mae']:.4f}")
    print(
        f"RMSE: {wind_results['metrics']['rmse']:.4f}")
    print(
        f"R²  : {wind_results['metrics']['r2_score']:.4f}")
    print("\nSolar Model")
    print(
        f"MAE : {solar_results['metrics']['mae']:.4f}"
    )
    print(
        f"RMSE: {solar_results['metrics']['rmse']:.4f}"
    )
    print(
        f"R²  : {solar_results['metrics']['r2_score']:.4f}"
    )
    print("\nSaved Files")
    print(
        f"Models      : {inc.XGB_MODEL_DIR}")
    print(
        f"Reports     : {inc.XGB_REPORT_DIR}"
    )
    print(
        f"Predictions : {prediction_path}"
    )
    print(
        f"Metrics     : {metrics_path}"
    )
    print(
        f"Report      : {report_path}"
    )

# ==========================================================
# Entry Point
# ==========================================================

if __name__ == "__main__":

    main()

