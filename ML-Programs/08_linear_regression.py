"""Linear Regression.

Extracted from FOML.ipynb.
Run this file with Python to reproduce the console output and plots.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

print('='*70)
print('LINEAR REGRESSION ALGORITHM')
print('='*70)

# Dataset: Hours Studied and Marks Obtained
X = np.array([[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]], dtype=float)
y = np.array([35,40,45,50,55,62,68,75,82,90], dtype=float)

print('\nDATASET')
print('-'*70)
print('Hours Studied\tMarks Obtained')
for hours, marks in zip(X.flatten(), y):
    print(hours, '\t\t', marks)

# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print('\nDATA SPLITTING')
print('-'*70)
print('Training samples:', len(X_train))
print('Testing samples :', len(X_test))

# Create Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

print('\nLINEAR REGRESSION MODEL')
print('-'*70)
print('Slope (Coefficient):', round(model.coef_[0], 4))
print('Intercept:', round(model.intercept_, 4))
print('Regression Equation: Marks =', round(model.coef_[0], 4), '* Hours +', round(model.intercept_, 4))

# Predict test data
y_pred = model.predict(X_test)

print('\nTEST PREDICTIONS')
print('-'*70)
print('Actual Marks\tPredicted Marks')
for actual, predicted in zip(y_test, y_pred):
    print(round(actual, 2), '\t\t', round(predicted, 2))

# Calculate performance metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print('\nMODEL PERFORMANCE')
print('-'*70)
print('Mean Squared Error (MSE)  :', round(mse, 4))
print('Root Mean Squared Error   :', round(rmse, 4))
print('R-squared (R2)            :', round(r2, 4))

# Predict marks for a new student
new_hours = np.array([[7.5]])
new_prediction = model.predict(new_hours)[0]

print('\nNEW SAMPLE PREDICTION')
print('-'*70)
print('Hours Studied:', new_hours[0][0])
print('Predicted Marks:', round(new_prediction, 2))

# Plot actual data and regression line
X_line = np.linspace(X.min(), X.max(), 100).reshape(-1,1)
y_line = model.predict(X_line)

plt.figure(figsize=(9,6))
plt.scatter(X, y, label='Actual Data')
plt.plot(X_line, y_line, linewidth=2, label='Regression Line')
plt.scatter(new_hours, new_prediction, s=100, marker='*', label='New Prediction')
plt.xlabel('Hours Studied')
plt.ylabel('Marks Obtained')
plt.title('Linear Regression - Hours Studied vs Marks')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot actual vs predicted values
plt.figure(figsize=(8,5))
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Marks')
plt.ylabel('Predicted Marks')
plt.title('Actual vs Predicted Marks')
plt.grid(True)
plt.tight_layout()
plt.show()

print('\n'+'='*70)
print('LINEAR REGRESSION IMPLEMENTATION COMPLETED SUCCESSFULLY')
print('='*70)

# ------------------------------------------------------------------
# SAVED CONSOLE OUTPUT (from the original notebook)
# ------------------------------------------------------------------
# This section is a reference; it is not executed as Python output.
# LINEAR REGRESSION ALGORITHM
# 
# DATASET
# ----------------------------------------------------------------------
# Hours Studied	Marks Obtained
# 1.0 		 35.0
# 2.0 		 40.0
# 3.0 		 45.0
# 4.0 		 50.0
# 5.0 		 55.0
# 6.0 		 62.0
# 7.0 		 68.0
# 8.0 		 75.0
# 9.0 		 82.0
# 10.0 		 90.0
# 
# DATA SPLITTING
# ----------------------------------------------------------------------
# Training samples: 8
# Testing samples : 2
# 
# LINEAR REGRESSION MODEL
# ----------------------------------------------------------------------
# Slope (Coefficient): 6.0862
# Intercept: 26.5259
# Regression Equation: Marks = 6.0862 * Hours + 26.5259
# 
# TEST PREDICTIONS
# ----------------------------------------------------------------------
# Actual Marks	Predicted Marks
# 82.0 		 81.3
# 40.0 		 38.7
# 
# MODEL PERFORMANCE
# ----------------------------------------------------------------------
# Mean Squared Error (MSE)  : 1.091
# Root Mean Squared Error   : 1.0445
# R-squared (R2)            : 0.9975
# 
# NEW SAMPLE PREDICTION
# ----------------------------------------------------------------------
# Hours Studied: 7.5
# Predicted Marks: 72.17
