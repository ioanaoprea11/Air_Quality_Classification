This project focuses on classifying air quality using Linear Discriminant Analysis (LDA) and Bayesian Discriminant Analysis. The dataset includes environmental factors that impact air quality, and the goal is to identify key determinants and create a predictive model.

📌 Features
Data Processing

Data sourced from Kaggle (WHO & World Bank).
Handling missing values by replacing them with column mean.
Splitting dataset: 70% training, 30% testing.
Air Quality Classification

Target variable: Air quality categories - "Good", "Moderate", "Poor", "Hazardous".
Predictors:
PM2.5, PM10 (particulate matter)
NO2, SO2, CO (pollutant gases)
Temperature, Humidity
Population Density
Proximity to Industrial Areas

Identified the most important predictors: CO, SO2, Proximity to Industrial Areas.
Computed discriminant scores and axes of separation.
Created scatter plots for visualizing class separation.
Evaluated model performance using confusion matrix, accuracy, and Cohen-Kappa index.

🛠️ Technologies Used
Python (NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn)
Statistical Modeling (LDA, Bayesian Classification)
Data Visualization (Scatter plots, Discriminant Analysis Projection)

📊 Results
Model	Overall Accuracy	Cohen-Kappa Index
LDA	92.83%	0.897
Bayesian	Lower than LDA	N/A
LDA showed better separation of air quality classes.
Final predictions saved in Predictii.csv.

