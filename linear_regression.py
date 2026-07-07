from pathlib import Path
import json
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import (
    ColumnTransformer,
    TransformedTargetRegressor,
)
from sklearn.feature_selection import (
    SelectPercentile,
    f_regression,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge  # Swapped from LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)

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

# PRUNED: Reduced from 30 to 10 to prevent feature avalanche
FEATURE_PERCENTILE = 10

# TAMED: Increased from 20 to 100 to group hundreds of unique locations
MIN_CATEGORY_FREQUENCY = 100


def cube_root(values):
    """Apply a cube-root transformation to the wind target."""
    return np.cbrt(values)


def cube_values(values):
    """Reverse the cube-root transformation."""
    return np.power(values, 3)


def load_shared_split():
    """Load and validate the shared chronological split."""
    required_files = [X_TRAIN_PATH, X_TEST_PATH, Y_TRAIN_PATH, Y_TEST_PATH]
    missing_files = [str(path) for path in required_files if not path.exists()]

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

    if len(X_train) != len(y_train) or len(X_test) != len(y_test):
        raise ValueError("X and y datasets have mismatched row counts.")

    return X_train, X_test, y_train, y_test


def prepare_model_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Prepare the modelling datasets."""
    if "datetime" in X_test.columns:
        test_datetimes = X_test["datetime"].copy()
    else:
        test_datetimes = pd.Series(range(len(X_test)), name="row_id")

    columns_to_drop = [col for col in ["datetime"] if col in X_train.columns]

    X_train_model = X_train.drop(columns=columns_to_drop).copy()
    X_test_model = X_test.drop(columns=columns_to_drop).copy()

    numeric_features = X_train_model.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = X_train_model.select_dtypes(include=["object", "string", "category"]).columns.tolist()

    print(f"Numeric features: {len(numeric_features)}")
    print(f"Categorical features: {len(categorical_features)}")

    return X_train_model, X_test_model, numeric_features, categorical_features, test_datetimes


def build_preprocessor(numeric_features, categorical_features):
    """Create preprocessing for numeric and categorical variables."""
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False)),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            min_frequency=MIN_CATEGORY_FREQUENCY,
            sparse_output=True,
        )),
    ])

    transformers = []
    if numeric_features:
        transformers.append(("numeric", numeric_pipeline, numeric_features))
    if categorical_features:
        transformers.append(("categorical", categorical_pipeline, categorical_features))

    return ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=1.0)


def build_regression_pipeline(numeric_features, categorical_features):
    """Build the complete Ridge regression pipeline."""
    preprocessor = build_preprocessor(numeric_features, categorical_features)

    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("feature_selection", SelectPercentile(
            score_func=f_regression,
            percentile=FEATURE_PERCENTILE,
        )),
        # REGULARIZED: Ridge applied with alpha=10.0 to penalize large coefficients
        ("model", Ridge(alpha=10.0)),
    ])


def calculate_metrics(y_true: pd.Series, y_pred: np.ndarray):
    """Calculate MAE, RMSE, and R²."""
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2_score": float(r2_score(y_true, y_pred)),
    }


def get_fitted_pipeline(trained_model):
    """Return the fitted Pipeline from either model type."""
    if isinstance(trained_model, TransformedTargetRegressor):
        return trained_model.regressor_
    return trained_model


def save_selected_features(trained_model, output_path: Path):
    """Save the feature names selected for the fitted model."""
    fitted_pipeline = get_fitted_pipeline(trained_model)
    preprocessor = fitted_pipeline.named_steps["preprocessor"]
    selector = fitted_pipeline.named_steps["feature_selection"]

    transformed_names = preprocessor.get_feature_names_out()
    selected_mask = selector.get_support()

    selected_features = pd.DataFrame({
        "feature": transformed_names[selected_mask],
        "f_score": selector.scores_[selected_mask],
    }).sort_values(by="f_score", ascending=False)

    selected_features.to_csv(output_path, index=False)
    return int(selected_mask.sum())


def main():
    """Run the updated Ridge Regression workflow."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = load_shared_split()

    X_train_model, X_test_model, numeric_features, categorical_features, test_datetimes = prepare_model_features(
        X_train, X_test
    )

    # Train the wind model
    print("\nTraining the wind power Ridge Regression model...")
    wind_pipeline = build_regression_pipeline(numeric_features, categorical_features)
    
    wind_model = TransformedTargetRegressor(
        regressor=wind_pipeline,
        func=cube_root,
        inverse_func=cube_values,
        check_inverse=False,
    )

    wind_model.fit(X_train_model, y_train[WIND_TARGET])
    wind_predictions = np.clip(wind_model.predict(X_test_model), a_min=0, a_max=None)
    wind_metrics = calculate_metrics(y_test[WIND_TARGET], wind_predictions)

    # Train the solar model
    print("Training the solar power Ridge Regression model...")
    solar_model = build_regression_pipeline(numeric_features, categorical_features)
    
    solar_model.fit(X_train_model, y_train[SOLAR_TARGET])
    solar_predictions = np.clip(solar_model.predict(X_test_model), a_min=0, a_max=None)
    solar_metrics = calculate_metrics(y_test[SOLAR_TARGET], solar_predictions)

    # Display results
    print("\nUpdated Ridge Regression workflow completed successfully.")
    
    print(f"\nWind power proxy results:")
    print(f"MAE: {wind_metrics['mae']:.4f}")
    print(f"RMSE: {wind_metrics['rmse']:.4f}")
    print(f"R² Score: {wind_metrics['r2_score']:.4f}")

    print(f"\nSolar power proxy results:")
    print(f"MAE: {solar_metrics['mae']:.4f}")
    print(f"RMSE: {solar_metrics['rmse']:.4f}")
    print(f"R² Score: {solar_metrics['r2_score']:.4f}")

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


def get_fitted_pipeline(trained_model):
    """Return the fitted Pipeline from either model type."""

    if isinstance(
        trained_model,
        TransformedTargetRegressor,
    ):
        return trained_model.regressor_

    return trained_model


def save_selected_features(
    trained_model,
    output_path: Path,
):
    """Save the feature names selected for the fitted model."""

    fitted_pipeline = get_fitted_pipeline(
        trained_model
    )

    preprocessor = fitted_pipeline.named_steps[
        "preprocessor"
    ]

    selector = fitted_pipeline.named_steps[
        "feature_selection"
    ]

    transformed_names = (
        preprocessor.get_feature_names_out()
    )

    selected_mask = selector.get_support()

    selected_features = pd.DataFrame(
        {
            "feature": transformed_names[
                selected_mask
            ],
            "f_score": selector.scores_[
                selected_mask
            ],
            "p_value": selector.pvalues_[
                selected_mask
            ],
        }
    ).sort_values(
        by="f_score",
        ascending=False,
    )

    selected_features.to_csv(
        output_path,
        index=False,
    )

    return int(
        selected_mask.sum()
    )


def save_coefficient_table(
    trained_model,
    selected_features_path: Path,
    output_path: Path,
):
    """Save coefficients matched to the selected feature names."""

    fitted_pipeline = get_fitted_pipeline(
        trained_model
    )

    regression_model = (
        fitted_pipeline.named_steps[
            "model"
        ]
    )

    selected_features = pd.read_csv(
        selected_features_path
    )

    coefficients = pd.DataFrame(
        {
            "feature": selected_features[
                "feature"
            ],
            "coefficient": (
                regression_model.coef_
            ),
            "absolute_coefficient": np.abs(
                regression_model.coef_
            ),
        }
    ).sort_values(
        by="absolute_coefficient",
        ascending=False,
    )

    coefficients.to_csv(
        output_path,
        index=False,
    )


def create_markdown_report(
    metrics,
):
    """Create the updated Member 4 report."""

    wind = metrics["wind_model"]
    solar = metrics["solar_model"]

    report = f"""# Member 4 – Linear Regression Report

## Objective

Train and evaluate separate Linear Regression models for wind and solar
renewable energy potential using the shared chronological train-test split.

## Dataset

- Training rows: {metrics['training_rows']:,}
- Testing rows: {metrics['testing_rows']:,}
- Numeric input columns: {metrics['numeric_feature_count']}
- Categorical input columns: {metrics['categorical_feature_count']}
- Split method: Strict chronological split
- Timestamp overlap: False

## Preprocessing

Numeric variables were handled using median imputation and standardization.

Categorical variables such as country, location, timezone, weather condition,
wind direction, and moon phase were retained and transformed using One-Hot
Encoding. Unknown categories in the test set were ignored safely, while rare
categories were grouped using a minimum-frequency threshold.

## Feature Selection

Univariate regression feature selection was performed separately for the wind
and solar models using `SelectPercentile` with `f_regression`.

- Feature percentile retained: {FEATURE_PERCENTILE}%
- Selected wind features: {wind['selected_feature_count']}
- Selected solar features: {solar['selected_feature_count']}

This allows the two models to retain different predictors based on their
individual relationships with wind and solar energy potential.

## Wind Target Transformation

The wind power proxy was created using wind speed cubed. A cube-root target
transformation was applied before training so that the model learned on the
underlying wind-speed scale. Model predictions were then cubed to return them
to the original wind power proxy scale.

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

- Wind and solar Linear Regression models
- Prediction file
- MAE, RMSE, and R² metrics
- Actual-versus-predicted plots
- Residual plots
- Selected-feature tables
- Coefficient tables
- JSON metrics file

## Conclusion

The updated models provide a stronger linear baseline by retaining categorical
information and selecting the most relevant transformed features separately
for wind and solar prediction.
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
    """Run the updated Linear Regression workflow."""

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
        X_train_model,
        X_test_model,
        numeric_features,
        categorical_features,
        test_datetimes,
    ) = prepare_model_features(
        X_train,
        X_test,
    )

    # Train the wind model:

    print(
        "\nTraining the wind power "
        "Linear Regression model..."
    )

    wind_pipeline = build_regression_pipeline(
        numeric_features,
        categorical_features,
    )

    wind_model = TransformedTargetRegressor(
        regressor=wind_pipeline,
        func=cube_root,
        inverse_func=cube_values,
        check_inverse=False,
    )

    wind_start_time = time.perf_counter()

    wind_model.fit(
        X_train_model,
        y_train[WIND_TARGET],
    )

    wind_training_time = (
        time.perf_counter()
        - wind_start_time
    )

    wind_predictions = wind_model.predict(
        X_test_model
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

    # Train the solar model:

    print(
        "Training the solar power "
        "Linear Regression model..."
    )

    solar_model = build_regression_pipeline(
        numeric_features,
        categorical_features,
    )

    solar_start_time = time.perf_counter()

    solar_model.fit(
        X_train_model,
        y_train[SOLAR_TARGET],
    )

    solar_training_time = (
        time.perf_counter()
        - solar_start_time
    )

    solar_predictions = solar_model.predict(
        X_test_model
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

    # Save models:

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

    save_actual_vs_predicted_plot(
        y_test[WIND_TARGET],
        wind_predictions,
        "Wind Power Proxy",
        REPORTS_DIR
        / "wind_actual_vs_predicted.png",
    )

    save_residual_plot(
        y_test[WIND_TARGET],
        wind_predictions,
        "Wind Power Proxy",
        REPORTS_DIR
        / "wind_residual_plot.png",
    )

    save_actual_vs_predicted_plot(
        y_test[SOLAR_TARGET],
        solar_predictions,
        "Solar Power Proxy",
        REPORTS_DIR
        / "solar_actual_vs_predicted.png",
    )

    save_residual_plot(
        y_test[SOLAR_TARGET],
        solar_predictions,
        "Solar Power Proxy",
        REPORTS_DIR
        / "solar_residual_plot.png",
    )

    # Save selected features and coefficients:

    wind_selected_path = (
        REPORTS_DIR
        / "wind_selected_features.csv"
    )

    solar_selected_path = (
        REPORTS_DIR
        / "solar_selected_features.csv"
    )

    wind_selected_count = save_selected_features(
        wind_model,
        wind_selected_path,
    )

    solar_selected_count = save_selected_features(
        solar_model,
        solar_selected_path,
    )

    save_coefficient_table(
        wind_model,
        wind_selected_path,
        REPORTS_DIR
        / "wind_feature_coefficients.csv",
    )

    save_coefficient_table(
        solar_model,
        solar_selected_path,
        REPORTS_DIR
        / "solar_feature_coefficients.csv",
    )

    # Save metrics:

    metrics = {
        "model_type": "Linear Regression",
        "split_method": (
            "strict_chronological"
        ),
        "training_rows": int(
            len(X_train_model)
        ),
        "testing_rows": int(
            len(X_test_model)
        ),
        "numeric_feature_count": int(
            len(numeric_features)
        ),
        "categorical_feature_count": int(
            len(categorical_features)
        ),
        "numeric_features": (
            numeric_features
        ),
        "categorical_features": (
            categorical_features
        ),
        "feature_selection_method": (
            "SelectPercentile(f_regression)"
        ),
        "feature_percentile": (
            FEATURE_PERCENTILE
        ),
        "minimum_category_frequency": (
            MIN_CATEGORY_FREQUENCY
        ),
        "wind_target_transformation": (
            "cube_root"
        ),
        "wind_model": {
            **wind_metrics,
            "training_time_seconds": float(
                wind_training_time
            ),
            "selected_feature_count": int(
                wind_selected_count
            ),
        },
        "solar_model": {
            **solar_metrics,
            "training_time_seconds": float(
                solar_training_time
            ),
            "selected_feature_count": int(
                solar_selected_count
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
        metrics
    )

    # Display results:

    print(
        "\nUpdated Linear Regression workflow "
        "completed successfully."
    )

    print("\nWind power proxy results:")
    print(
        f"MAE:"
        f"{wind_metrics['mae']:.4f}"
    )
    print(
        f"RMSE:"
        f"{wind_metrics['rmse']:.4f}"
    )
    print(
        f"R² Score: "
        f"{wind_metrics['r2_score']:.4f}"
    )
    print(
        f"Selected features: "
        f"{wind_selected_count}"
    )

    print("\nSolar power proxy results:")
    print(
        f"MAE:"
        f"{solar_metrics['mae']:.4f}"
    )
    print(
        f"RMSE:"
        f"{solar_metrics['rmse']:.4f}"
    )
    print(
        f"R² Score: "
        f"{solar_metrics['r2_score']:.4f}"
    )
    print(
        f"Selected features: "
        f"{solar_selected_count}"
    )

    print("\nSaved files:")
    print(f"Wind model: {wind_model_path}")
    print(f"Solar model: {solar_model_path}")
    print(f"Predictions: {predictions_path}")
    print(f"Metrics: {metrics_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
