"""K-Nearest Neighbours (K-NN) Classification.

Extracted from FOML.ipynb.
Run this file with Python to reproduce the console output and plots.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

print('='*70)
print('K-NEAREST NEIGHBOURS (K-NN) CLASSIFICATION')
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

# Display first 10 samples
print('\nFIRST 10 SAMPLES')
print('-'*70)
for i in range(10):
    print(X[i], '->', iris.target_names[y[i]])

# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print('\nDATA SPLITTING')
print('-'*70)
print('Training samples:', len(X_train))
print('Testing samples :', len(X_test))

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create K-NN classifier
k = 5
knn = KNeighborsClassifier(n_neighbors=k)

# Train the model
knn.fit(X_train, y_train)

print('\nK-NN MODEL')
print('-'*70)
print('Value of K:', k)
print('Distance Metric: Euclidean Distance')

# Prediction
predictions = knn.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)
print('\nMODEL ACCURACY')
print('-'*70)
print('Accuracy:', round(accuracy * 100, 2), '%')

# Display predictions
print('\nPREDICTIONS')
print('-'*70)
print('Actual Class\tPredicted Class')
for actual, predicted in zip(y_test, predictions):
    print(iris.target_names[actual], '\t', iris.target_names[predicted])

# Classification report
print('\nCLASSIFICATION REPORT')
print('-'*70)
print(classification_report(y_test, predictions, target_names=iris.target_names))

# Confusion matrix
cm = confusion_matrix(y_test, predictions)
print('\nCONFUSION MATRIX')
print('-'*70)
print(cm)

# Classify a new sample
new_sample = np.array([[5.1, 3.5, 1.4, 0.2]])
new_sample_scaled = scaler.transform(new_sample)
new_prediction = knn.predict(new_sample_scaled)[0]
new_probabilities = knn.predict_proba(new_sample_scaled)[0]

print('\nNEW SAMPLE CLASSIFICATION')
print('-'*70)
print('New sample:', new_sample[0])
print('Predicted class:', iris.target_names[new_prediction])
print('Class probabilities:')
for name, probability in zip(iris.target_names, new_probabilities):
    print(name, ':', round(probability * 100, 2), '%')

# Graph 1: Confusion Matrix
plt.figure(figsize=(7,5))
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names).plot()
plt.title('K-NN Confusion Matrix')
plt.tight_layout()
plt.show()

# Graph 2: Accuracy for different K values
k_values = range(1, 16)
accuracies = []

for value in k_values:
    model = KNeighborsClassifier(n_neighbors=value)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    accuracies.append(accuracy_score(y_test, pred))

plt.figure(figsize=(9,5))
plt.plot(k_values, accuracies, 'o-')
plt.xlabel('Value of K')
plt.ylabel('Accuracy')
plt.title('K-NN Accuracy for Different K Values')
plt.xticks(list(k_values))
plt.grid(True)
plt.tight_layout()
plt.show()

print('\n'+'='*70)
print('K-NN IMPLEMENTATION COMPLETED SUCCESSFULLY')
print('='*70)

# ------------------------------------------------------------------
# SAVED CONSOLE OUTPUT (from the original notebook)
# ------------------------------------------------------------------
# This section is a reference; it is not executed as Python output.
# K-NEAREST NEIGHBOURS (K-NN) CLASSIFICATION
# 
# DATASET INFORMATION
# ----------------------------------------------------------------------
# Dataset       : Iris Dataset
# Total Samples : 150
# Features      : ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
# Classes       : ['setosa' 'versicolor' 'virginica']
# 
# FIRST 10 SAMPLES
# ----------------------------------------------------------------------
# [5.1 3.5 1.4 0.2] -> setosa
# [4.9 3.  1.4 0.2] -> setosa
# [4.7 3.2 1.3 0.2] -> setosa
# [4.6 3.1 1.5 0.2] -> setosa
# [5.  3.6 1.4 0.2] -> setosa
# [5.4 3.9 1.7 0.4] -> setosa
# [4.6 3.4 1.4 0.3] -> setosa
# [5.  3.4 1.5 0.2] -> setosa
# [4.4 2.9 1.4 0.2] -> setosa
# [4.9 3.1 1.5 0.1] -> setosa
# 
# DATA SPLITTING
# ----------------------------------------------------------------------
# Training samples: 120
# Testing samples : 30
# 
# K-NN MODEL
# ----------------------------------------------------------------------
# Value of K: 5
# Distance Metric: Euclidean Distance
# 
# MODEL ACCURACY
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
# virginica 	 versicolor
# setosa 	 setosa
# virginica 	 virginica
# versicolor 	 versicolor
# virginica 	 versicolor
# virginica 	 virginica
# versicolor 	 versicolor
# versicolor 	 versicolor
# setosa 	 setosa
# virginica 	 virginica
# setosa 	 setosa
# 
# CLASSIFICATION REPORT
# ----------------------------------------------------------------------
#               precision    recall  f1-score   support
# 
#       setosa       1.00      1.00      1.00        10
#   versicolor       0.83      1.00      0.91        10
#    virginica       1.00      0.80      0.89        10
# 
#     accuracy                           0.93        30
#    macro avg       0.94      0.93      0.93        30
# weighted avg       0.94      0.93      0.93        30
# 
# 
# CONFUSION MATRIX
# ----------------------------------------------------------------------
# [[10  0  0]
#  [ 0 10  0]
#  [ 0  2  8]]
# 
# NEW SAMPLE CLASSIFICATION
# ----------------------------------------------------------------------
# New sample: [5.1 3.5 1.4 0.2]
# Predicted class: setosa
# Class probabilities:
# setosa : 100.0 %
# versicolor : 0.0 %
# virginica : 0.0 %
