# RenewableEnergyOutputForecasting
# Member 1: Data Preprocessing
Load the Global Weather Repository dataset.
Handle missing values and duplicates.
Detect and treat outliers.
Convert date/time columns to the correct format.
Standardize units and data types.
Produce the cleaned dataset for the rest of the team.
# Member 2: Feature Engineering
Create the Wind Power Proxy using wind speed.
Create the Solar Power Proxy using UV Index, cloud cover, and time features.
Generate lag features (1-hour, 6-hour, and 24-hour).
Generate rolling statistics (24-hour mean and standard deviation).
Create cyclical time features (hour/day using sine and cosine).
Deliver the final feature-engineered dataset.
# Member 3: Exploratory Data Analysis (EDA)
Explore the dataset and identify patterns.
Analyze relationships between weather variables.
Create visualizations such as:
Correlation heatmap
Histograms
Box plots
Line charts
Scatter plots
Summarize key insights from the data.
Member 4: Linear Regression Model
Train a Linear Regression model to predict renewable energy generation potential.
Tune model parameters if necessary.
Evaluate the model using:
MAE
RMSE
R² Score
Save predictions and evaluation results.
# Member 5: Decision Tree Regressor
Train a Decision Tree Regressor.
Perform hyperparameter tuning.
Evaluate using:
MAE
RMSE
R² Score
Save predictions and evaluation results.
# Member 6: Random Forest Regressor
Train a Random Forest Regressor.
Optimize model parameters.
Evaluate using:
MAE
RMSE
R² Score
Save predictions and evaluation results.
# Member 7: XGBoost Regressor
Train an XGBoost Regressor.
Tune hyperparameters.
Evaluate using:
MAE
RMSE
R² Score
Compare its performance with the other three models.
Save predictions and evaluation results.
# Member 8: Model Evaluation, Explainability & Dashboard
Compare the performance of all four models using MAE, RMSE, and R².
Select the best-performing model.
Generate feature importance and SHAP explainability plots for the selected model.
Build the final dashboard (e.g., using Streamlit or Power BI) to display:
Predicted wind energy potential
Predicted solar energy potential
Overall renewable energy potential
Key weather variables
Model performance metrics
Prepare the final visualizations for the presentation.
