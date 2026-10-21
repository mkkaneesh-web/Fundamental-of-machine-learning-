"""Logistic Regression Classification.

Extracted from FOML.ipynb.
Run this file with Python to reproduce the console output and plots.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

print('='*70)
print('LOGISTIC REGRESSION CLASSIFICATION')
print('='*70)

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

print('\nDATASET INFORMATION')
print('-'*70)
print('Dataset       : Iris Dataset')
print('Total Samples :', X.shape[0])
print('Features      :', iris.feature_names)
print('Classes       :', iris.target_names)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print('\nDATA SPLITTING')
print('-'*70)
print('Training samples:', len(X_train))
print('Testing samples :', len(X_test))

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create Logistic Regression model
model = LogisticRegression(max_iter=1000, random_state=42)

# Train the model
model.fit(X_train, y_train)

print('\nLOGISTIC REGRESSION MODEL')
print('-'*70)
print('Model trained successfully.')
print('Solver: lbfgs')

# Predict test data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print('\nACCURACY')
print('-'*70)
print('Accuracy:', round(accuracy * 100, 2), '%')

# Display predictions
print('\nPREDICTIONS')
print('-'*70)
print('Actual Class\tPredicted Class')
for actual, predicted in zip(y_test, y_pred):
    print(iris.target_names[actual], '\t', iris.target_names[predicted])

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

print('\nCONFUSION MATRIX')
print('-'*70)
print(cm)

# Classification report
print('\nCLASSIFICATION REPORT')
print('-'*70)
print(classification_report(y_test, y_pred, target_names=iris.target_names, zero_division=0))

# Classify a new sample
new_sample = np.array([[5.1, 3.5, 1.4, 0.2]])
new_sample_scaled = scaler.transform(new_sample)
new_prediction = model.predict(new_sample_scaled)[0]
probabilities = model.predict_proba(new_sample_scaled)[0]

print('\nNEW SAMPLE CLASSIFICATION')
print('-'*70)
print('New sample:', new_sample[0])
print('Predicted class:', iris.target_names[new_prediction])
print('\nClass probabilities:')
for name, probability in zip(iris.target_names, probabilities):
    print(name, ':', round(probability * 100, 2), '%')

# Plot confusion matrix
plt.figure(figsize=(7,5))
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names).plot()
plt.title('Logistic Regression - Confusion Matrix')
plt.tight_layout()
plt.show()

# Accuracy graph
plt.figure(figsize=(6,5))
plt.bar(['Logistic Regression'], [accuracy * 100])
plt.ylabel('Accuracy (%)')
plt.title('Logistic Regression Accuracy')
plt.ylim(0, 100)
plt.grid(axis='y')
plt.tight_layout()
plt.show()

print('\n'+'='*70)
print('LOGISTIC REGRESSION IMPLEMENTATION COMPLETED SUCCESSFULLY')
print('='*70)

# ------------------------------------------------------------------
# SAVED CONSOLE OUTPUT (from the original notebook)
# ------------------------------------------------------------------
# This section is a reference; it is not executed as Python output.
# LOGISTIC REGRESSION CLASSIFICATION
# 
# DATASET INFORMATION
# ----------------------------------------------------------------------
# Dataset       : Iris Dataset
# Total Samples : 150
# Features      : ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
# Classes       : ['setosa' 'versicolor' 'virginica']
# 
# DATA SPLITTING
# ----------------------------------------------------------------------
# Training samples: 120
# Testing samples : 30
# 
# LOGISTIC REGRESSION MODEL
# ----------------------------------------------------------------------
# Model trained successfully.
# Solver: lbfgs
# 
# ACCURACY
# ----------------------------------------------------------------------
# Accuracy: 93.33 %
# 
# PREDICTIONS
# ----------------------------------------------------------------------
# Actual Class	Predicted Class
# setosa 	 setosa
# virginica 	 virginica
# versicolor 	 versicolor
# versicolor 	 versicolor
# setosa 	 setosa
# versicolor 	 versicolor
# setosa 	 setosa
# setosa 	 setosa
# virginica 	 virginica
# versicolor 	 versicolor
# virginica 	 virginica
# virginica 	 virginica
# virginica 	 virginica
# versicolor 	 versicolor
# setosa 	 setosa
# setosa 	 setosa
# setosa 	 setosa
# versicolor 	 versicolor
# versicolor 	 versicolor
# virginica 	 virginica
# setosa 	 setosa
# virginica 	 virginica
# versicolor 	 versicolor
# virginica 	 versicolor
# virginica 	 virginica
# versicolor 	 virginica
# versicolor 	 versicolor
# setosa 	 setosa
# virginica 	 virginica
# setosa 	 setosa
# 
# CONFUSION MATRIX
# ----------------------------------------------------------------------
# [[10  0  0]
#  [ 0  9  1]
#  [ 0  1  9]]
# 
# CLASSIFICATION REPORT
# ----------------------------------------------------------------------
#               precision    recall  f1-score   support
# 
#       setosa       1.00      1.00      1.00        10
#   versicolor       0.90      0.90      0.90        10
#    virginica       0.90      0.90      0.90        10
# 
#     accuracy                           0.93        30
#    macro avg       0.93      0.93      0.93        30
# weighted avg       0.93      0.93      0.93        30
# 
# 
# NEW SAMPLE CLASSIFICATION
# ----------------------------------------------------------------------
# New sample: [5.1 3.5 1.4 0.2]
# Predicted class: setosa
# 
# Class probabilities:
# setosa : 98.08 %
# versicolor : 1.92 %
# virginica : 0.0 %
