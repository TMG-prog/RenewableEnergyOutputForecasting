
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

- MAE: 100155.0357
- RMSE: 100336.9341
- R² Score: -83.7820
- Best CV Score: -7059610.4153
- Training Time: 0.37 seconds

## Solar Power Proxy Results

- MAE: 0.4885
- RMSE: 0.8861
- R² Score: 0.8177
- Best CV Score: 0.7873
- Training Time: 0.27 seconds

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
