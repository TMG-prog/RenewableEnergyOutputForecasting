"""
Member 6: Random Forest Regressor models for renewable energy potential.

This script:
1. Loads the shared chronological train-test split.
2. Selects numeric predictor columns.
3. Prepares a Pipeline with median imputation, standard scaling, and Random Forest Regressor.
4. Performs hyperparameter tuning using RandomizedSearchCV (on a 20% sample of training data)
   with parameters aimed at model compression (limited depth, restricted leaf nodes).
5. Fits the final models (with best parameters) on the full training dataset.
6. Applies a cube-root target transformation to the wind target.
7. Evaluates both models using MAE, RMSE, and R².
8. Saves trained models, predictions, feature importances, plots, metrics, and a Markdown report.
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
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
    / "random_forest"
)

REPORTS_DIR = (
    PROJECT_DIR
    / "reports"
    / "random_forest"
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
        raise ValueError("X_train and y_train have different row counts.")

    if len(X_test) != len(y_test):
        raise ValueError("X_test and y_test have different row counts.")

    required_targets = [WIND_TARGET, SOLAR_TARGET]

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
        raise ValueError("The training targets contain missing values.")

    if y_test[required_targets].isnull().any().any():
        raise ValueError("The testing targets contain missing values.")

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape:  {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape:  {y_test.shape}")

    return X_train, X_test, y_train, y_test


def prepare_numeric_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Select numeric predictor columns."""
    if "datetime" in X_test.columns:
        test_datetimes = X_test["datetime"].copy()
    else:
        test_datetimes = pd.Series(
            range(len(X_test)),
            name="row_id",
        )

    numeric_features = X_train.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()

    if not numeric_features:
        raise ValueError("No numeric predictor columns were found.")

    missing_in_test = [
        column
        for column in numeric_features
        if column not in X_test.columns
    ]

    if missing_in_test:
        raise ValueError(
            "The following training features are missing from X_test:\n"
            + "\n".join(missing_in_test)
        )

    X_train_numeric = X_train[numeric_features].copy()
    X_test_numeric = X_test[numeric_features].copy()

    print(f"Numeric features selected: {len(numeric_features)}")

    return (
        X_train_numeric,
        X_test_numeric,
        numeric_features,
        test_datetimes,
    )


def build_base_pipeline():
    """Create a preprocessing and Random Forest pipeline."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                RandomForestRegressor(random_state=42, n_jobs=-1),
            ),
        ]
    )


def calculate_metrics(y_true: pd.Series, y_pred: np.ndarray):
    """Calculate MAE, RMSE, and R²."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

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
    sample_size = min(5000, len(y_true))
    sample_indices = np.linspace(
        0,
        len(y_true) - 1,
        sample_size,
        dtype=int,
    )

    actual_values = np.asarray(y_true)[sample_indices]
    predicted_values = np.asarray(y_pred)[sample_indices]

    minimum = min(actual_values.min(), predicted_values.min())
    maximum = max(actual_values.max(), predicted_values.max())

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.scatter(actual_values, predicted_values, alpha=0.35, s=14)
    ax.plot(
        [minimum, maximum],
        [minimum, maximum],
        linestyle="--",
        linewidth=2,
        label="Perfect prediction",
    )

    ax.set_xlabel("Actual value")
    ax.set_ylabel("Predicted value")
    ax.set_title(f"Actual vs Predicted — {target_name}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_residual_plot(
    y_true: pd.Series,
    y_pred: np.ndarray,
    target_name: str,
    output_path: Path,
):
    """Save a residual plot."""
    actual_values = np.asarray(y_true)
    predicted_values = np.asarray(y_pred)
    residuals = actual_values - predicted_values

    sample_size = min(5000, len(y_true))
    sample_indices = np.linspace(
        0, len(y_true) - 1, sample_size, dtype=int
    )

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.scatter(
        predicted_values[sample_indices],
        residuals[sample_indices],
        alpha=0.35,
        s=14,
    )
    ax.axhline(y=0, linestyle="--", linewidth=2)
    ax.set_xlabel("Predicted value")
    ax.set_ylabel("Residual")
    ax.set_title(f"Residual Plot — {target_name}")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_importance_table(
    trained_model,
    feature_names,
    output_path: Path,
):
    """Extract and save the feature importances from the model."""
    if isinstance(trained_model, TransformedTargetRegressor):
        fitted_pipeline = trained_model.regressor_
    else:
        fitted_pipeline = trained_model

    rf_model = fitted_pipeline.named_steps["model"]

    importances = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": rf_model.feature_importances_,
        }
    )
    importances = importances.sort_values(by="importance", ascending=False)
    importances.to_csv(output_path, index=False)
    return importances


def tune_model(
    X_train_numeric: pd.DataFrame,
    y_train_target: pd.Series,
    target_name: str,
    is_wind: bool = False,
):
    """
    Tune Random Forest hyperparameters using a 20% chronological sub-sample
    of the training data to ensure quick, efficient runs, while strictly
    optimizing for compressed and shallow trees (max depth bounds).
    """
    print(f"\n--- Hyperparameter Tuning for {target_name} ---")

    n_samples = len(X_train_numeric)
    sample_size = int(n_samples * 0.20)
    
    X_tune = X_train_numeric.iloc[-sample_size:].copy()
    y_tune = y_train_target.iloc[-sample_size:].copy()
    
    print(f"Tuning on 20% chronological subset: {X_tune.shape} rows.")

    # Parameters targeted for compression:
    param_distributions = {
        "model__n_estimators": [30, 50, 80],
        "model__max_depth": [6, 10, 14],
        "model__min_samples_split": [5, 10],
        "model__min_samples_leaf": [4, 8, 16],
        "model__max_features": ["sqrt", "log2"],
    }

    base_pipeline = build_base_pipeline()

    if is_wind:
        tuning_estimator = TransformedTargetRegressor(
            regressor=base_pipeline,
            func=cube_root,
            inverse_func=cube_values,
            check_inverse=False,
        )
        tuned_params = {
            f"regressor__{k}": v for k, v in param_distributions.items()
        }
    else:
        tuning_estimator = base_pipeline
        tuned_params = param_distributions

    tscv = TimeSeriesSplit(n_splits=3)

    search = RandomizedSearchCV(
        estimator=tuning_estimator,
        param_distributions=tuned_params,
        n_iter=5,
        scoring="neg_mean_absolute_error",
        cv=tscv,
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )

    start_time = time.perf_counter()
    search.fit(X_tune, y_tune)
    tuning_time = time.perf_counter() - start_time

    print(f"Tuning completed in {tuning_time:.2f} seconds.")
    print("Best parameters found:")
    
    best_params = search.best_params_
    for param, val in best_params.items():
        print(f"  {param}: {val}")

    return best_params


def train_final_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    best_params: dict,
    target_name: str,
    is_wind: bool = False,
):
    """Train the final Random Forest Regressor on the full training dataset using best params."""
    print(f"\nTraining final {target_name} model on full training set...")

    pipeline = build_base_pipeline()
    
    if is_wind:
        clean_params = {}
        for k, v in best_params.items():
            if k.startswith("regressor__"):
                clean_params[k.replace("regressor__", "")] = v
            else:
                clean_params[k] = v
        
        model_params = {k.replace("model__", ""): v for k, v in clean_params.items() if k.startswith("model__")}
        pipeline.named_steps["model"].set_params(**model_params)
        
        final_model = TransformedTargetRegressor(
            regressor=pipeline,
            func=cube_root,
            inverse_func=cube_values,
            check_inverse=False,
        )
    else:
        model_params = {k.replace("model__", ""): v for k, v in best_params.items() if k.startswith("model__")}
        pipeline.named_steps["model"].set_params(**model_params)
        final_model = pipeline

    start_time = time.perf_counter()
    final_model.fit(X_train, y_train)
    training_time = time.perf_counter() - start_time
    print(f"Model training completed in {training_time:.2f} seconds.")

    return final_model, training_time


def create_markdown_report(
    metrics,
    feature_count,
    training_rows,
    testing_rows,
):
    """Create the Member 6 Random Forest report."""
    wind = metrics["wind_model"]
    solar = metrics["solar_model"]

    report = f"""# Member 6 – Random Forest Regressor Report

## Objective

Train, tune, and evaluate compressed Random Forest Regressor models for renewable energy generation
potential using the shared chronological train-test split.

Two separate models were developed:
- Wind power proxy prediction
- Solar power proxy prediction

## Model Compression & Optimization Details

To keep the models lightweight (compressed) and limit memory usage, we optimized hyperparameters with constraints:
- **Maximum Depth**: Capped at explicit depths (e.g., 6, 10, or 14) instead of unrestricted depth (`None`).
- **Estimators**: Restrained to 30, 50, or 80 trees to minimize memory footprint.
- **Min Samples per Leaf**: Set to 4, 8, or 16 to reduce structural complexity.
- **Feature Selection at Split**: Restricted to square root (`sqrt`) or binary logarithm (`log2`) of features.

Hyperparameter tuning was completed using `RandomizedSearchCV` and a 3-fold `TimeSeriesSplit` on a 20% chronological sample of the training data. The final models were fitted on the full dataset using the discovered best parameters.

### Wind Power Proxy Optimized Parameters
- Estimators: {wind['best_params'].get('regressor__model__n_estimators', wind['best_params'].get('model__n_estimators'))}
- Max Depth: {wind['best_params'].get('regressor__model__max_depth', wind['best_params'].get('model__max_depth'))}
- Min Samples Split: {wind['best_params'].get('regressor__model__min_samples_split', wind['best_params'].get('model__min_samples_split'))}
- Min Samples Leaf: {wind['best_params'].get('regressor__model__min_samples_leaf', wind['best_params'].get('model__min_samples_leaf'))}
- Max Features: {wind['best_params'].get('regressor__model__max_features', wind['best_params'].get('model__max_features'))}

### Solar Power Proxy Optimized Parameters
- Estimators: {solar['best_params'].get('model__n_estimators')}
- Max Depth: {solar['best_params'].get('model__max_depth')}
- Min Samples Split: {solar['best_params'].get('model__min_samples_split')}
- Min Samples Leaf: {solar['best_params'].get('model__min_samples_leaf')}
- Max Features: {solar['best_params'].get('model__max_features')}

## Dataset

- Training rows: {training_rows:,}
- Testing rows: {testing_rows:,}
- Numeric predictor features: {feature_count}
- Split method: Strict chronological split
- Timestamp overlap: False

## Modelling Approach

Numeric predictor columns were preprocessed using median imputation and standardized before training.

The wind power proxy model applies a cube-root target transformation via `TransformedTargetRegressor` to match the Linear Regression pipeline, converting predictions back to the original wind-power proxy scale before evaluation.
The solar power proxy model is fitted directly. Negative predictions were clipped to zero for both models.

## Wind Power Proxy Results
- **MAE**: {wind['mae']:.4f}
- **RMSE**: {wind['rmse']:.4f}
- **R² Score**: {wind['r2_score']:.4f}
- **Training time**: {wind['training_time_seconds']:.2f} seconds

## Solar Power Proxy Results
- **MAE**: {solar['mae']:.4f}
- **RMSE**: {solar['rmse']:.4f}
- **R² Score**: {solar['r2_score']:.4f}
- **Training time**: {solar['training_time_seconds']:.2f} seconds

## Generated Deliverables
- Wind Random Forest model (compressed)
- Solar Random Forest model (compressed)
- Prediction file (`random_forest_predictions.csv`)
- Actual-versus-predicted plots
- Residual plots
- Feature importance tables
- JSON metrics file

## Conclusion
The compressed Random Forest Regressors capture non-linear relationships in weather predictors while maintaining a very low memory footprint and fast inference times. 
"""

    report_path = REPORTS_DIR / "member6_random_forest_report.md"
    report_path.write_text(report, encoding="utf-8")
    return report_path


def main():
    """Run the complete Random Forest workflow."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = load_shared_split()

    X_train_numeric, X_test_numeric, feature_names, test_datetimes = prepare_numeric_features(
        X_train, X_test
    )

    # 1. Tune and Train Wind Model:
    best_wind_params = tune_model(
        X_train_numeric,
        y_train[WIND_TARGET],
        "Wind Power Proxy",
        is_wind=True,
    )

    wind_model, wind_training_time = train_final_model(
        X_train_numeric,
        y_train[WIND_TARGET],
        best_wind_params,
        "Wind Power Proxy",
        is_wind=True,
    )

    wind_predictions = wind_model.predict(X_test_numeric)
    wind_predictions = np.clip(wind_predictions, a_min=0, a_max=None)
    wind_metrics = calculate_metrics(y_test[WIND_TARGET], wind_predictions)

    # 2. Tune and Train Solar Model:
    best_solar_params = tune_model(
        X_train_numeric,
        y_train[SOLAR_TARGET],
        "Solar Power Proxy",
        is_wind=False,
    )

    solar_model, solar_training_time = train_final_model(
        X_train_numeric,
        y_train[SOLAR_TARGET],
        best_solar_params,
        "Solar Power Proxy",
        is_wind=False,
    )

    solar_predictions = solar_model.predict(X_test_numeric)
    solar_predictions = np.clip(solar_predictions, a_min=0, a_max=None)
    solar_metrics = calculate_metrics(y_test[SOLAR_TARGET], solar_predictions)

    # Save trained models:
    wind_model_path = MODELS_DIR / "random_forest_wind.pkl"
    solar_model_path = MODELS_DIR / "random_forest_solar.pkl"

    print("\nSaving compressed trained models...")
    joblib.dump(wind_model, wind_model_path, compress=3)
    joblib.dump(solar_model, solar_model_path, compress=3)
    print("Models saved successfully.")

    # Save predictions:
    predictions = pd.DataFrame(
        {
            "datetime": test_datetimes.reset_index(drop=True),
            "actual_wind_power_proxy": y_test[WIND_TARGET].reset_index(drop=True),
            "predicted_wind_power_proxy": wind_predictions,
            "actual_solar_power_proxy": y_test[SOLAR_TARGET].reset_index(drop=True),
            "predicted_solar_power_proxy": solar_predictions,
        }
    )

    predictions_path = REPORTS_DIR / "random_forest_predictions.csv"
    predictions.to_csv(predictions_path, index=False)

    # Save plots:
    wind_actual_plot = REPORTS_DIR / "wind_actual_vs_predicted.png"
    wind_residual_plot = REPORTS_DIR / "wind_residual_plot.png"
    solar_actual_plot = REPORTS_DIR / "solar_actual_vs_predicted.png"
    solar_residual_plot = REPORTS_DIR / "solar_residual_plot.png"

    print("Generating and saving evaluation plots...")
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

    # Save feature importances:
    wind_importance_path = REPORTS_DIR / "wind_feature_importances.csv"
    solar_importance_path = REPORTS_DIR / "solar_feature_importances.csv"

    save_importance_table(wind_model, feature_names, wind_importance_path)
    save_importance_table(solar_model, feature_names, solar_importance_path)

    # Save metrics JSON:
    metrics = {
        "model_type": "Random Forest Regressor",
        "split_method": "strict_chronological",
        "training_rows": int(len(X_train_numeric)),
        "testing_rows": int(len(X_test_numeric)),
        "numeric_feature_count": int(len(feature_names)),
        "numeric_features": feature_names,
        "wind_target_transformation": "cube_root",
        "wind_model": {
            **wind_metrics,
            "training_time_seconds": float(wind_training_time),
            "best_params": best_wind_params,
        },
        "solar_model": {
            **solar_metrics,
            "training_time_seconds": float(solar_training_time),
            "best_params": best_solar_params,
        },
    }

    metrics_path = REPORTS_DIR / "random_forest_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=4)

    # Save report:
    report_path = create_markdown_report(
        metrics,
        len(feature_names),
        len(X_train_numeric),
        len(X_test_numeric),
    )

    print("\nRandom Forest workflow completed successfully.")
    print("\nWind power proxy results:")
    print(f"  MAE:      {wind_metrics['mae']:.4f}")
    print(f"  RMSE:     {wind_metrics['rmse']:.4f}")
    print(f"  R² Score: {wind_metrics['r2_score']:.4f}")
    print("\nSolar power proxy results:")
    print(f"  MAE:      {solar_metrics['mae']:.4f}")
    print(f"  RMSE:     {solar_metrics['rmse']:.4f}")
    print(f"  R² Score: {solar_metrics['r2_score']:.4f}")

    print("\nSaved files:")
    print(f"  Wind model:  {wind_model_path}")
    print(f"  Solar model: {solar_model_path}")
    print(f"  Predictions: {predictions_path}")
    print(f"  Metrics:     {metrics_path}")
    print(f"  Report:      {report_path}")


if __name__ == "__main__":
    main()
