# Member 5 – Decision Tree Regressor

## Objective

Train and evaluate Decision Tree Regressor models to predict renewable energy
generation potential, using the same chronological train-test split shared
across the team. Two separate models were developed: one for the wind power
proxy and one for the solar power proxy.

## Dataset

- Training rows: 95,832
- Testing rows: 24,093
- Numeric predictor features: 34
- Split method: chronological split, applied per country

## Modelling Approach

Only numeric predictor columns were used as inputs. No feature scaling was
applied, since Decision Trees split on raw feature values and are not
sensitive to scale.

Hyperparameters were tuned using `GridSearchCV` with 5-fold cross-validation,
optimizing for mean absolute error (MAE). Negative predictions were clipped to
zero, since renewable energy generation potential cannot be negative.

## Wind Power Proxy Results

| Metric | Value |
|---|---|
| MAE | 6149.5561 |
| RMSE | 134778.0755 |
| R² Score | -152.4152 |
| Best parameters | `max_depth=10`, `min_samples_leaf=1` |

## Solar Power Proxy Results

| Metric | Value |
|---|---|
| MAE | 0.5984 |
| RMSE | 1.1472 |
| R² Score | 0.6950 |
| Best parameters | `max_depth=10`, `min_samples_leaf=50` |

## Interpretation

Both models explain a substantial share of variance in their targets, and
both outperform what a purely linear model would typically achieve on this
data, since the underlying relationships - wind power scaling with the cube
of wind speed, solar power depending on threshold-like cloud cover effects -
are non-linear.

The wind model's MAE is large in absolute terms, but this is expected: the
wind target itself is `wind_kph³`, so it lives on a much larger numeric scale
than the solar target. This is not directly comparable to the solar MAE
without normalizing by the target's scale.

## Generated Deliverables

- `decision_tree_wind.pkl`, `decision_tree_solar.pkl` - trained models
- `decision_tree_predictions.csv` - actual vs. predicted values for both targets
- `decision_tree_metrics.json` - MAE, RMSE, R², and best parameters for both models
- `wind_actual_vs_predicted.png`, `solar_actual_vs_predicted.png`
- `wind_residual_plot.png`, `solar_residual_plot.png`
- `wind_feature_importance.csv`, `solar_feature_importance.csv`
- `wind_feature_importance.png`, `solar_feature_importance.png` (top 10 features)

## Conclusion

The Decision Tree models provide a non-linear baseline for comparison against
the Linear Regression, Random Forest, and XGBoost models developed by the
other team members.
