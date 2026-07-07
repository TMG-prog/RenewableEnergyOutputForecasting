
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

- MAE: 3392.9869
- RMSE: 9665.2986
- R² Score: 0.2110
- Best CV Score: 0.4396
- Training Time: 0.93 seconds

## Solar Power Proxy Results

- MAE: 0.5853
- RMSE: 1.0709
- R² Score: 0.7342
- Best CV Score: 0.6880
- Training Time: 0.73 seconds

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
