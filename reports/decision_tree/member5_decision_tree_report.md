# Member 5 – Decision Tree Regressor Report

## Objective

Train and evaluate Decision Tree Regressor models to predict renewable energy
generation potential, using the same chronological train-test split shared
across the team. Two separate models were developed: one for the wind power
proxy and one for the solar power proxy.

## Dataset

- Training rows: 96,107
- Testing rows: 24,041
- Numeric predictor features: 34
- Split method: strict chronological split (same split used by Member 4's
  Linear Regression model, so results are directly comparable)

## Modelling Approach

Only numeric predictor columns were used as inputs, matching the baseline set
by the Linear Regression model. No feature scaling was applied, since Decision
Trees split on raw feature values and are not sensitive to scale — this is one
place the tree-based approach is simpler than Linear Regression.

Hyperparameters were tuned using `GridSearchCV` with 5-fold cross-validation,
optimizing for mean absolute error (MAE). The search covered:

- `max_depth`: 5, 10, or unlimited
- `min_samples_leaf`: 1, 5, 10, 20, 30, 50

The `min_samples_leaf` grid was deliberately widened partway through
development after the solar model's first result landed on the edge of a
narrower grid (20), which signaled the search wanted more regularization than
it was offered. Re-running with the wider grid moved the chosen value to an
interior point (30), confirming a genuine optimum rather than an
artificially grid-limited one, and improved the solar model's R² from 0.75 to
0.78.

Negative predictions were clipped to zero, since renewable energy generation
potential cannot be negative.

## Wind Power Proxy Results

| Metric | Value |
|---|---|
| MAE | 1373.0947 |
| RMSE | 4179.2252 |
| R² Score | 0.8534 |
| Best parameters | `max_depth=10`, `min_samples_leaf=5` |

## Solar Power Proxy Results

| Metric | Value |
|---|---|
| MAE | 0.4936 |
| RMSE | 0.9547 |
| R² Score | 0.7839 |
| Best parameters | `max_depth=10`, `min_samples_leaf=30` |

## Interpretation

Both models explain a substantial share of variance in their targets (R² of
0.85 for wind, 0.78 for solar), and both outperform what a purely linear model
would typically achieve on this data, since the underlying relationships —
wind power scaling with the cube of wind speed, solar power depending on
threshold-like cloud cover effects — are non-linear.

The wind model's MAE is large in absolute terms, but this is expected: the
wind target itself is `wind_kph³`, so it lives on a much larger numeric scale
than the solar target. This is not directly comparable to the solar MAE
without normalizing by the target's scale.

## Generated Deliverables

- `decision_tree_wind.pkl`, `decision_tree_solar.pkl` — trained models
- `decision_tree_predictions.csv` — actual vs. predicted values for both targets
- `decision_tree_metrics.json` — MAE, RMSE, R², and best parameters for both models
- `wind_actual_vs_predicted.png`, `solar_actual_vs_predicted.png`
- `wind_residual_plot.png`, `solar_residual_plot.png`
- `wind_feature_importance.csv`, `solar_feature_importance.csv`
- `wind_feature_importance.png`, `solar_feature_importance.png` (top 10 features)

## Conclusion

The Decision Tree models provide a non-linear baseline for comparison against
the Linear Regression, Random Forest, and XGBoost models developed by the
other team members. Their stronger fit relative to the linear baseline
suggests non-linear relationships in the weather-to-power mapping that the
later ensemble models (Random Forest, XGBoost) are likely to capture even
more effectively.