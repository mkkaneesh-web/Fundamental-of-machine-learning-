"""Comparison of Linear and Polynomial Regression.

Extracted from FOML.ipynb.
Run this file with Python to reproduce the console output and plots.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score

print('='*75)
print('COMPARISON OF LINEAR AND POLYNOMIAL REGRESSION')
print('='*75)

# Dataset: Hours Studied vs Marks
X = np.array([[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]], dtype=float)
y = np.array([35,40,45,52,60,70,78,85,91,95], dtype=float)

print('\nDATASET')
print('-'*75)
print('Hours Studied\tMarks Obtained')
for hours, marks in zip(X.flatten(), y):
    print(hours, '\t\t', marks)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print('\nDATA SPLITTING')
print('-'*75)
print('Training samples:', len(X_train))
print('Testing samples :', len(X_test))

# ---------------- LINEAR REGRESSION ----------------
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)

linear_mse = mean_squared_error(y_test, y_pred_linear)
linear_rmse = np.sqrt(linear_mse)
linear_r2 = r2_score(y_test, y_pred_linear)

# ---------------- POLYNOMIAL REGRESSION ----------------
degree = 2
polynomial_model = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression())
polynomial_model.fit(X_train, y_train)
y_pred_poly = polynomial_model.predict(X_test)

poly_mse = mean_squared_error(y_test, y_pred_poly)
poly_rmse = np.sqrt(poly_mse)
poly_r2 = r2_score(y_test, y_pred_poly)

# Display results
print('\nLINEAR REGRESSION RESULTS')
print('-'*75)
print('Slope:', round(linear_model.coef_[0], 4))
print('Intercept:', round(linear_model.intercept_, 4))
print('MSE :', round(linear_mse, 4))
print('RMSE:', round(linear_rmse, 4))
print('R2  :', round(linear_r2, 4))

print('\nPOLYNOMIAL REGRESSION RESULTS')
print('-'*75)
print('Polynomial Degree:', degree)
print('MSE :', round(poly_mse, 4))
print('RMSE:', round(poly_rmse, 4))
print('R2  :', round(poly_r2, 4))

# Prediction for new sample
new_hours = np.array([[7.5]])
linear_prediction = linear_model.predict(new_hours)[0]
poly_prediction = polynomial_model.predict(new_hours)[0]

print('\nNEW SAMPLE PREDICTION')
print('-'*75)
print('Hours Studied:', new_hours[0][0])
print('Linear Regression Prediction   :', round(linear_prediction, 2))
print('Polynomial Regression Prediction:', round(poly_prediction, 2))

# Comparison table
print('\nMODEL COMPARISON')
print('-'*75)
print('Model\t\t\tMSE\t\tRMSE\t\tR2')
print('Linear Regression\t', round(linear_mse,4), '\t', round(linear_rmse,4), '\t', round(linear_r2,4))
print('Polynomial Regression\t', round(poly_mse,4), '\t', round(poly_rmse,4), '\t', round(poly_r2,4))

# Determine better model
if poly_r2 > linear_r2:
    print('\nBetter Model: Polynomial Regression')
else:
    print('\nBetter Model: Linear Regression')

# Generate smooth curves
X_curve = np.linspace(1, 10, 200).reshape(-1,1)
y_linear_curve = linear_model.predict(X_curve)
y_poly_curve = polynomial_model.predict(X_curve)

# Plot comparison
plt.figure(figsize=(10,6))
plt.scatter(X, y, label='Actual Data')
plt.plot(X_curve, y_linear_curve, linewidth=2, label='Linear Regression')
plt.plot(X_curve, y_poly_curve, linewidth=2, label='Polynomial Regression (Degree 2)')
plt.xlabel('Hours Studied')
plt.ylabel('Marks Obtained')
plt.title('Linear vs Polynomial Regression')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# R2 comparison graph
plt.figure(figsize=(8,5))
models = ['Linear Regression', 'Polynomial Regression']
r2_values = [linear_r2, poly_r2]
plt.bar(models, r2_values)
plt.ylabel('R2 Score')
plt.title('R2 Score Comparison')
plt.ylim(0, 1.1)
plt.grid(axis='y')
plt.tight_layout()
plt.show()

print('\n'+'='*75)
print('LINEAR AND POLYNOMIAL REGRESSION COMPARISON COMPLETED')
print('='*75)

# ------------------------------------------------------------------
# SAVED CONSOLE OUTPUT (from the original notebook)
# ------------------------------------------------------------------
# This section is a reference; it is not executed as Python output.
# COMPARISON OF LINEAR AND POLYNOMIAL REGRESSION
# 
# DATASET
# ---------------------------------------------------------------------------
# Hours Studied	Marks Obtained
# 1.0 		 35.0
# 2.0 		 40.0
# 3.0 		 45.0
# 4.0 		 52.0
# 5.0 		 60.0
# 6.0 		 70.0
# 7.0 		 78.0
# 8.0 		 85.0
# 9.0 		 91.0
# 10.0 		 95.0
# 
# DATA SPLITTING
# ---------------------------------------------------------------------------
# Training samples: 8
# Testing samples : 2
# 
# LINEAR REGRESSION RESULTS
# ---------------------------------------------------------------------------
# Slope: 7.1379
# Intercept: 25.7414
# MSE : 0.5175
# RMSE: 0.7194
# R2  : 0.9992
# 
# POLYNOMIAL REGRESSION RESULTS
# ---------------------------------------------------------------------------
# Polynomial Degree: 2
# MSE : 0.5175
# RMSE: 0.7194
# R2  : 0.9992
# 
# NEW SAMPLE PREDICTION
# ---------------------------------------------------------------------------
# Hours Studied: 7.5
# Linear Regression Prediction   : 79.28
# Polynomial Regression Prediction: 79.28
# 
# MODEL COMPARISON
# ---------------------------------------------------------------------------
# Model			MSE		RMSE		R2
# Linear Regression	 0.5175 	 0.7194 	 0.9992
# Polynomial Regression	 0.5175 	 0.7194 	 0.9992
# 
# Better Model: Linear Regression
# 
# 
# LINEAR AND POLYNOMIAL REGRESSION COMPARISON COMPLETED
