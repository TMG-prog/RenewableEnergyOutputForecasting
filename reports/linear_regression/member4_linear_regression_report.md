# Member 4 – Linear Regression Report

## Objective

Train and evaluate Linear Regression models for renewable energy generation
potential using the shared chronological train-test split.

Two separate models were developed:

- Wind power proxy prediction
- Solar power proxy prediction

## Dataset

- Training rows: 96,107
- Testing rows: 24,041
- Numeric predictor features: 34
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

- MAE: 1764.1866
- RMSE: 6093.9817
- R² Score: 0.6883
- Training time: 0.67 seconds

## Solar Power Proxy Results

- MAE: 0.6324
- RMSE: 1.0620
- R² Score: 0.7326
- Training time: 0.54 seconds

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
