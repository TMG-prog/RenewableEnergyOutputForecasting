# Member 4 – Linear Regression Report

## Objective

Train and evaluate separate Linear Regression models for wind and solar
renewable energy potential using the shared chronological train-test split.

## Dataset

- Training rows: 96,107
- Testing rows: 24,041
- Numeric input columns: 34
- Categorical input columns: 10
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

- Feature percentile retained: 30%
- Selected wind features: 1268
- Selected solar features: 1268

This allows the two models to retain different predictors based on their
individual relationships with wind and solar energy potential.

## Wind Target Transformation

The wind power proxy was created using wind speed cubed. A cube-root target
transformation was applied before training so that the model learned on the
underlying wind-speed scale. Model predictions were then cubed to return them
to the original wind power proxy scale.

## Wind Power Proxy Results

- MAE: 1755.2329
- RMSE: 6175.7861
- R² Score: 0.6799
- Training time: 7.67 seconds

## Solar Power Proxy Results

- MAE: 0.5653
- RMSE: 0.9656
- R² Score: 0.7790
- Training time: 6.68 seconds

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
