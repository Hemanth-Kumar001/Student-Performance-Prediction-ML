# Student Academic Performance Prediction

A regression project that estimates a student's final score from five academic performance attributes. It compares Linear Regression and Random Forest Regression on an 80/20 train-test split.

## Dataset

File: Student_Performance.csv  
Rows: 600  
Target: Final_Score

Input columns:
- Study_Hours
- Attendance
- Assignment_Score
- Internal_Marks
- Previous_Score

The uploaded CSV has no missing values.

## Algorithms

- Linear Regression
- Random Forest Regressor (n_estimators=200)

Median imputation is included in each model pipeline.

## Evaluation

The script reports Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared (R2). It saves a comparison table and an actual-versus-predicted scatter plot.

## Run locally

Install dependencies with: python -m pip install -r requirements.txt

Run: python student_performance_prediction.py

Generated output files:
- student_regression_comparison.csv
- student_actual_vs_predicted.png

## Notes

- The script uses random_state=42 for reproducibility.
- No performance values are claimed here; run the script to generate results in your environment.
- Dataset source and redistribution terms were not provided with the uploaded file. Confirm the source/license before redistributing the CSV elsewhere.
