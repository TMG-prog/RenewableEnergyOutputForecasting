# Member 6 – Random Forest Regressor Report

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
- Estimators: 80
- Max Depth: 14
- Min Samples Split: 10
- Min Samples Leaf: 8
- Max Features: sqrt

### Solar Power Proxy Optimized Parameters
- Estimators: 80
- Max Depth: 10
- Min Samples Split: 5
- Min Samples Leaf: 8
- Max Features: log2

## Dataset

- Training rows: 95,832
- Testing rows: 24,093
- Numeric predictor features: 34
- Split method: Strict chronological split
- Timestamp overlap: False

## Modelling Approach

Numeric predictor columns were preprocessed using median imputation and standardized before training.

The wind power proxy model applies a cube-root target transformation via `TransformedTargetRegressor` to match the Linear Regression pipeline, converting predictions back to the original wind-power proxy scale before evaluation.
The solar power proxy model is fitted directly. Negative predictions were clipped to zero for both models.

## Wind Power Proxy Results
- **MAE**: 3439.2879
- **RMSE**: 9426.9117
- **R² Score**: 0.2495
- **Training time**: 5.18 seconds

## Solar Power Proxy Results
- **MAE**: 0.6312
- **RMSE**: 1.0771
- **R² Score**: 0.7311
- **Training time**: 3.27 seconds

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
