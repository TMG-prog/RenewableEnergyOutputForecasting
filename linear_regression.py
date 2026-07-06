"""
Member 4: Linear Regression models for renewable energy potential.

This script:
1. Loads the shared chronological train-test split.
2. Selects numeric predictor columns.
3. Trains separate Linear Regression models for wind and solar power proxies.
4. Applies a cube-root transformation to the wind target.
5. Evaluates both models using MAE, RMSE, and R².
6. Saves trained models, predictions, plots, metrics, and a Markdown report.
"""

from pathlib import Path
import json
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# Configuration:

PROJECT_DIR = Path(__file__).resolve().parent

DATA_DIR = (
    PROJECT_DIR
    / "Project"
    / "data"
    / "train"
)

MODELS_DIR = (
    PROJECT_DIR
    / "models"
    / "linear_regression"
)

REPORTS_DIR = (
    PROJECT_DIR
    / "reports"
    / "linear_regression"
)

X_TRAIN_PATH = DATA_DIR / "X_train.csv"
X_TEST_PATH = DATA_DIR / "X_test.csv"
Y_TRAIN_PATH = DATA_DIR / "y_train.csv"
Y_TEST_PATH = DATA_DIR / "y_test.csv"

WIND_TARGET = "wind_power_proxy"
SOLAR_TARGET = "solar_power_proxy"


def cube_root(values):
    """Apply a cube-root transformation."""

    return np.cbrt(values)


def cube_values(values):
    """Reverse the cube-root transformation."""

    return np.power(values, 3)


def load_shared_split():
    """Load and validate the shared train-test files."""

    required_files = [
        X_TRAIN_PATH,
        X_TEST_PATH,
        Y_TRAIN_PATH,
        Y_TEST_PATH,
    ]

    missing_files = [
        str(path)
        for path in required_files
        if not path.exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            "The following required files were not found:\n"
            + "\n".join(missing_files)
        )

    print("Loading the shared train-test split...")

    X_train = pd.read_csv(X_TRAIN_PATH)
    X_test = pd.read_csv(X_TEST_PATH)
    y_train = pd.read_csv(Y_TRAIN_PATH)
    y_test = pd.read_csv(Y_TEST_PATH)

    if len(X_train) != len(y_train):
        raise ValueError(
            "X_train and y_train have different row counts."
        )

    if len(X_test) != len(y_test):
        raise ValueError(
            "X_test and y_test have different row counts."
        )

    required_targets = [
        WIND_TARGET,
        SOLAR_TARGET,
    ]

    missing_targets = [
        target
        for target in required_targets
        if target not in y_train.columns
        or target not in y_test.columns
    ]

    if missing_targets:
        raise ValueError(
            "The following target columns are missing:\n"
            + "\n".join(missing_targets)
        )

    if y_train[required_targets].isnull().any().any():
        raise ValueError(
            "The training targets contain missing values."
        )

    if y_test[required_targets].isnull().any().any():
        raise ValueError(
            "The testing targets contain missing values."
        )

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape:  {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape:  {y_test.shape}")

    return X_train, X_test, y_train, y_test


def prepare_numeric_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
):
    """
    Select numeric predictor columns.

    Text columns such as country, location, weather descriptions,
    directions, and moon phases are excluded from this baseline model.
    """

    if "datetime" in X_test.columns:
        test_datetimes = X_test[
            "datetime"
        ].copy()
    else:
        test_datetimes = pd.Series(
            range(len(X_test)),
            name="row_id",
        )

    numeric_features = X_train.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()

    if not numeric_features:
        raise ValueError(
            "No numeric predictor columns were found."
        )

    missing_in_test = [
        column
        for column in numeric_features
        if column not in X_test.columns
    ]

    if missing_in_test:
        raise ValueError(
            "The following training features are missing "
            "from X_test:\n"
            + "\n".join(missing_in_test)
        )

    X_train_numeric = X_train[
        numeric_features
    ].copy()

    X_test_numeric = X_test[
        numeric_features
    ].copy()

    print(
        f"Numeric features selected: "
        f"{len(numeric_features)}"
    )

    return (
        X_train_numeric,
        X_test_numeric,
        numeric_features,
        test_datetimes,
    )


def build_base_pipeline():
    """Create a preprocessing and Linear Regression pipeline."""

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                LinearRegression(),
            ),
        ]
    )


def calculate_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
):
    """Calculate MAE, RMSE, and R²."""

    mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    rmse = np.sqrt(
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


def save_actual_vs_predicted_plot(
    y_true: pd.Series,
    y_pred: np.ndarray,
    target_name: str,
    output_path: Path,
):
    """Save an actual-versus-predicted scatter plot."""

    sample_size = min(
        5000,
        len(y_true),
    )

    sample_indices = np.linspace(
        0,
        len(y_true) - 1,
        sample_size,
        dtype=int,
    )

    actual_values = np.asarray(
        y_true
    )[sample_indices]

    predicted_values = np.asarray(
        y_pred
    )[sample_indices]

    minimum = min(
        actual_values.min(),
        predicted_values.min(),
    )

    maximum = max(
        actual_values.max(),
        predicted_values.max(),
    )

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.scatter(
        actual_values,
        predicted_values,
        alpha=0.35,
        s=14,
    )

    ax.plot(
        [minimum, maximum],
        [minimum, maximum],
        linestyle="--",
        linewidth=2,
        label="Perfect prediction",
    )

    ax.set_xlabel("Actual value")
    ax.set_ylabel("Predicted value")

    ax.set_title(
        f"Actual vs Predicted — {target_name}"
    )

    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def save_residual_plot(
    y_true: pd.Series,
    y_pred: np.ndarray,
    target_name: str,
    output_path: Path,
):
    """Save a residual plot."""

    actual_values = np.asarray(
        y_true
    )

    predicted_values = np.asarray(
        y_pred
    )

    residuals = (
        actual_values
        - predicted_values
    )

    sample_size = min(
        5000,
        len(y_true),
    )

    sample_indices = np.linspace(
        0,
        len(y_true) - 1,
        sample_size,
        dtype=int,
    )

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.scatter(
        predicted_values[sample_indices],
        residuals[sample_indices],
        alpha=0.35,
        s=14,
    )

    ax.axhline(
        y=0,
        linestyle="--",
        linewidth=2,
    )

    ax.set_xlabel("Predicted value")
    ax.set_ylabel("Residual")

    ax.set_title(
        f"Residual Plot — {target_name}"
    )

    ax.grid(alpha=0.3)

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def save_coefficient_table(
    trained_model,
    feature_names,
    output_path: Path,
):
    """Save the Linear Regression feature coefficients."""

    if isinstance(
        trained_model,
        TransformedTargetRegressor,
    ):
        fitted_pipeline = (
            trained_model.regressor_
        )
    else:
        fitted_pipeline = trained_model

    regression_model = (
        fitted_pipeline.named_steps[
            "model"
        ]
    )

    coefficients = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": regression_model.coef_,
            "absolute_coefficient": np.abs(
                regression_model.coef_
            ),
        }
    )

    coefficients = coefficients.sort_values(
        by="absolute_coefficient",
        ascending=False,
    )

    coefficients.to_csv(
        output_path,
        index=False,
    )


def create_markdown_report(
    metrics,
    feature_count,
    training_rows,
    testing_rows,
):
    """Create the Member 4 Linear Regression report."""

    wind = metrics["wind_model"]
    solar = metrics["solar_model"]

    report = f"""# Member 4 – Linear Regression Report

## Objective

Train and evaluate Linear Regression models for renewable energy generation
potential using the shared chronological train-test split.

Two separate models were developed:

- Wind power proxy prediction
- Solar power proxy prediction

## Dataset

- Training rows: {training_rows:,}
- Testing rows: {testing_rows:,}
- Numeric predictor features: {feature_count}
- Split method: Strict chronological split
- Timestamp overlap: False

## Modelling Approach

Only numeric predictor columns were used in this baseline model. Missing
numeric values were handled using median imputation, and all predictors were
standardized before training.

The wind power proxy was calculated as the cube of wind speed. A cube-root
target transformation was therefore applied during training. Predictions were
converted back to the original wind-power proxy scale before evaluation.

The solar power proxy was modelled directly using standard Linear Regression.

Negative predictions were clipped to zero because renewable energy generation
potential cannot be negative.

## Wind Power Proxy Results

- MAE: {wind['mae']:.4f}
- RMSE: {wind['rmse']:.4f}
- R² Score: {wind['r2_score']:.4f}
- Training time: {wind['training_time_seconds']:.2f} seconds

## Solar Power Proxy Results

- MAE: {solar['mae']:.4f}
- RMSE: {solar['rmse']:.4f}
- R² Score: {solar['r2_score']:.4f}
- Training time: {solar['training_time_seconds']:.2f} seconds

## Generated Deliverables

- Wind Linear Regression model
- Solar Linear Regression model
- Prediction file
- MAE, RMSE, and R² metrics
- Actual-versus-predicted plots
- Residual plots
- Feature coefficient tables
- JSON metrics file

## Conclusion

The two models provide a transparent linear baseline for comparison with the
Decision Tree, Random Forest, and XGBoost regression models developed by the
other team members.
"""

    report_path = (
        REPORTS_DIR
        / "member4_linear_regression_report.md"
    )

    report_path.write_text(
        report,
        encoding="utf-8",
    )

    return report_path


def main():
    """Run the complete Linear Regression workflow."""

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = load_shared_split()

    (
        X_train_numeric,
        X_test_numeric,
        feature_names,
        test_datetimes,
    ) = prepare_numeric_features(
        X_train,
        X_test,
    )

    # Train the wind power model:

    print(
        "\nTraining the wind power "
        "Linear Regression model..."
    )

    wind_pipeline = (
        build_base_pipeline()
    )

    wind_model = (
        TransformedTargetRegressor(
            regressor=wind_pipeline,
            func=cube_root,
            inverse_func=cube_values,
            check_inverse=False,
        )
    )

    wind_start_time = (
        time.perf_counter()
    )

    wind_model.fit(
        X_train_numeric,
        y_train[WIND_TARGET],
    )

    wind_training_time = (
        time.perf_counter()
        - wind_start_time
    )

    wind_predictions = wind_model.predict(
        X_test_numeric
    )

    wind_predictions = np.clip(
        wind_predictions,
        a_min=0,
        a_max=None,
    )

    wind_metrics = calculate_metrics(
        y_test[WIND_TARGET],
        wind_predictions,
    )

    # Train the solar power model:

    print(
        "Training the solar power "
        "Linear Regression model..."
    )

    solar_model = (
        build_base_pipeline()
    )

    solar_start_time = (
        time.perf_counter()
    )

    solar_model.fit(
        X_train_numeric,
        y_train[SOLAR_TARGET],
    )

    solar_training_time = (
        time.perf_counter()
        - solar_start_time
    )

    solar_predictions = solar_model.predict(
        X_test_numeric
    )

    solar_predictions = np.clip(
        solar_predictions,
        a_min=0,
        a_max=None,
    )

    solar_metrics = calculate_metrics(
        y_test[SOLAR_TARGET],
        solar_predictions,
    )

    # Save trained models:

    wind_model_path = (
        MODELS_DIR
        / "linear_regression_wind.pkl"
    )

    solar_model_path = (
        MODELS_DIR
        / "linear_regression_solar.pkl"
    )

    joblib.dump(
        wind_model,
        wind_model_path,
    )

    joblib.dump(
        solar_model,
        solar_model_path,
    )

    # Save predictions:

    predictions = pd.DataFrame(
        {
            "datetime": (
                test_datetimes
                .reset_index(drop=True)
            ),
            "actual_wind_power_proxy": (
                y_test[WIND_TARGET]
                .reset_index(drop=True)
            ),
            "predicted_wind_power_proxy": (
                wind_predictions
            ),
            "actual_solar_power_proxy": (
                y_test[SOLAR_TARGET]
                .reset_index(drop=True)
            ),
            "predicted_solar_power_proxy": (
                solar_predictions
            ),
        }
    )

    predictions_path = (
        REPORTS_DIR
        / "linear_regression_predictions.csv"
    )

    predictions.to_csv(
        predictions_path,
        index=False,
    )

    # Save plots:

    wind_actual_plot = (
        REPORTS_DIR
        / "wind_actual_vs_predicted.png"
    )

    wind_residual_plot = (
        REPORTS_DIR
        / "wind_residual_plot.png"
    )

    solar_actual_plot = (
        REPORTS_DIR
        / "solar_actual_vs_predicted.png"
    )

    solar_residual_plot = (
        REPORTS_DIR
        / "solar_residual_plot.png"
    )

    save_actual_vs_predicted_plot(
        y_test[WIND_TARGET],
        wind_predictions,
        "Wind Power Proxy",
        wind_actual_plot,
    )

    save_residual_plot(
        y_test[WIND_TARGET],
        wind_predictions,
        "Wind Power Proxy",
        wind_residual_plot,
    )

    save_actual_vs_predicted_plot(
        y_test[SOLAR_TARGET],
        solar_predictions,
        "Solar Power Proxy",
        solar_actual_plot,
    )

    save_residual_plot(
        y_test[SOLAR_TARGET],
        solar_predictions,
        "Solar Power Proxy",
        solar_residual_plot,
    )

    # Save coefficient tables:

    wind_coefficients_path = (
        REPORTS_DIR
        / "wind_feature_coefficients.csv"
    )

    solar_coefficients_path = (
        REPORTS_DIR
        / "solar_feature_coefficients.csv"
    )

    save_coefficient_table(
        wind_model,
        feature_names,
        wind_coefficients_path,
    )

    save_coefficient_table(
        solar_model,
        feature_names,
        solar_coefficients_path,
    )

    # Save metrics:

    metrics = {
        "model_type": "Linear Regression",
        "split_method": (
            "strict_chronological"
        ),
        "training_rows": int(
            len(X_train_numeric)
        ),
        "testing_rows": int(
            len(X_test_numeric)
        ),
        "numeric_feature_count": int(
            len(feature_names)
        ),
        "numeric_features": (
            feature_names
        ),
        "wind_target_transformation": (
            "cube_root"
        ),
        "wind_model": {
            **wind_metrics,
            "training_time_seconds": float(
                wind_training_time
            ),
        },
        "solar_model": {
            **solar_metrics,
            "training_time_seconds": float(
                solar_training_time
            ),
        },
    }

    metrics_path = (
        REPORTS_DIR
        / "linear_regression_metrics.json"
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8",
    ) as metrics_file:
        json.dump(
            metrics,
            metrics_file,
            indent=4,
        )

    report_path = create_markdown_report(
        metrics,
        len(feature_names),
        len(X_train_numeric),
        len(X_test_numeric),
    )

    # Display results:

    print(
        "\nLinear Regression workflow "
        "completed successfully."
    )

    print("\nWind power proxy results:")

    print(
        f"MAE:      "
        f"{wind_metrics['mae']:.4f}"
    )

    print(
        f"RMSE:     "
        f"{wind_metrics['rmse']:.4f}"
    )

    print(
        f"R² Score: "
        f"{wind_metrics['r2_score']:.4f}"
    )

    print("\nSolar power proxy results:")

    print(
        f"MAE:      "
        f"{solar_metrics['mae']:.4f}"
    )

    print(
        f"RMSE:     "
        f"{solar_metrics['rmse']:.4f}"
    )

    print(
        f"R² Score: "
        f"{solar_metrics['r2_score']:.4f}"
    )

    print("\nSaved files:")

    print(
        f"Wind model: "
        f"{wind_model_path}"
    )

    print(
        f"Solar model: "
        f"{solar_model_path}"
    )

    print(
        f"Predictions: "
        f"{predictions_path}"
    )

    print(
        f"Metrics: "
        f"{metrics_path}"
    )

    print(
        f"Report: "
        f"{report_path}"
    )


if __name__ == "__main__":
    main()