# Cardiovascular-Disease-Prediction
This project focuses on building a machine learning model to analyze cardiovascular health data and predict outcomes based on patient lifestyle, medical history, and health indicators.
The workflow includes data preprocessing, feature engineering, exploratory data analysis (EDA), and model training using a Random Forest Classifier.

📂 Project Structure

Cardiovascular (1).ipynb – Main Jupyter notebook containing the full workflow.

CVD_cleaned.csv – Dataset used in this analysis (cleaned version of cardiovascular health survey data).

⚙️ Workflow Overview

Importing Libraries

Numpy, Pandas, Matplotlib, Seaborn for data handling and visualization.

Scikit-learn for preprocessing, model training, and evaluation.

Data Loading & Exploration

Load dataset (CVD_cleaned.csv).

Inspect shape, missing values, summary statistics.

Data Cleaning & Encoding

Converted categorical features (e.g., General Health, Checkup, Diabetes, Exercise, Skin Cancer, etc.) into numeric values.

Handled categorical variables with OneHotEncoder where necessary.

Exploratory Data Analysis (EDA)

Distribution plots and correlations to understand relationships between risk factors and cardiovascular outcomes.

Feature Engineering

Defined independent variables (X) and target variable (y).

Split data into training and testing sets.

Model Training & Evaluation

Built a Random Forest Classifier.

Performed predictions on the test set.

Evaluated performance using accuracy and classification metrics.

Tuned hyperparameters with RandomizedSearchCV for optimization.

📊 Key Techniques Used

Data Encoding: replace(), OneHotEncoder for categorical variables.

Visualization: Seaborn & Matplotlib (heatmaps, distributions).

Machine Learning: Random Forest Classifier for prediction.

Hyperparameter Tuning: RandomizedSearchCV to improve model performance.


✅ Results

The Random Forest model successfully predicts cardiovascular disease risk.

Hyperparameter tuning improved accuracy and robustness.

Feature importance analysis highlights key risk factors contributing to cardiovascular health outcomes.

📌 Future Improvements

Test additional models (Logistic Regression, XGBoost, Neural Networks).

Improve handling of class imbalance.

Deploy as a web app (using Flask or Streamlit) for interactive predictions.
