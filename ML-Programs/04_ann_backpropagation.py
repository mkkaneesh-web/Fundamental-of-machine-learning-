"""Artificial Neural Network Using Backpropagation.

Extracted from FOML.ipynb.
Run this file with Python to reproduce the console output and plots.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print('='*70)
print('ARTIFICIAL NEURAL NETWORK USING BACKPROPAGATION')
print('='*70)

# XOR dataset - suitable for demonstrating a neural network
X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
y = np.array([[0],[1],[1],[0]], dtype=float)

print('\nDATASET')
print('-'*70)
for i in range(len(X)):
    print('Input:', X[i], 'Target:', int(y[i][0]))

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative(x):
    return x * (1 - x)

# Random initialization
np.random.seed(42)

input_neurons = 2
hidden_neurons = 4
output_neurons = 1
learning_rate = 0.5
epochs = 10000

W1 = np.random.uniform(-1, 1, (input_neurons, hidden_neurons))
b1 = np.zeros((1, hidden_neurons))
W2 = np.random.uniform(-1, 1, (hidden_neurons, output_neurons))
b2 = np.zeros((1, output_neurons))

loss_history = []

print('\nNETWORK ARCHITECTURE')
print('-'*70)
print('Input Layer  : 2 neurons')
print('Hidden Layer : 4 neurons')
print('Output Layer : 1 neuron')
print('Activation   : Sigmoid')
print('Learning Rate:', learning_rate)
print('Epochs       :', epochs)

# Backpropagation training
for epoch in range(epochs):
    # Forward propagation
    hidden_input = np.dot(X, W1) + b1
    hidden_output = sigmoid(hidden_input)

    final_input = np.dot(hidden_output, W2) + b2
    final_output = sigmoid(final_input)

    # Mean squared error
    error = y - final_output
    loss = np.mean(error ** 2)
    loss_history.append(loss)

    # Backpropagation
    output_delta = error * sigmoid_derivative(final_output)
    hidden_error = np.dot(output_delta, W2.T)
    hidden_delta = hidden_error * sigmoid_derivative(hidden_output)

    # Update weights and biases
    W2 += learning_rate * np.dot(hidden_output.T, output_delta)
    b2 += learning_rate * np.sum(output_delta, axis=0, keepdims=True)

    W1 += learning_rate * np.dot(X.T, hidden_delta)
    b1 += learning_rate * np.sum(hidden_delta, axis=0, keepdims=True)

    if (epoch + 1) % 1000 == 0:
        print('Epoch:', epoch + 1, 'Loss:', round(loss, 6))

# Testing
hidden_output = sigmoid(np.dot(X, W1) + b1)
final_output = sigmoid(np.dot(hidden_output, W2) + b2)
predictions = (final_output >= 0.5).astype(int)

print('\nFINAL RESULTS')
print('-'*70)
print('Input\tTarget\tOutput\tPrediction')
for i in range(len(X)):
    print(X[i], '\t', int(y[i][0]), '\t', round(float(final_output[i][0]), 4), '\t', int(predictions[i][0]))

accuracy = accuracy_score(y, predictions)
print('\nAccuracy:', round(accuracy * 100, 2), '%')

print('\nCONFUSION MATRIX')
print('-'*70)
print(confusion_matrix(y, predictions))

print('\nCLASSIFICATION REPORT')
print('-'*70)
print(classification_report(y, predictions, zero_division=0))

print('\nTRAINED WEIGHTS')
print('-'*70)
print('Input -> Hidden weights:')
print(W1)
print('\nHidden -> Output weights:')
print(W2)

# Graph 1: Loss curve
plt.figure(figsize=(9,5))
plt.plot(loss_history)
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.title('Backpropagation Training Loss')
plt.grid(True)
plt.tight_layout()
plt.show()

# Graph 2: Actual vs predicted
plt.figure(figsize=(8,5))
plt.plot(range(len(y)), y.flatten(), 'o-', label='Actual')
plt.plot(range(len(predictions)), predictions.flatten(), 's--', label='Predicted')
plt.xlabel('Sample')
plt.ylabel('Class')
plt.title('Actual vs Predicted Output')
plt.xticks(range(len(y)))
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print('\n'+'='*70)
print('ANN BACKPROPAGATION DEMONSTRATION COMPLETED')
print('='*70)

# ------------------------------------------------------------------
# SAVED CONSOLE OUTPUT (from the original notebook)
# ------------------------------------------------------------------
# This section is a reference; it is not executed as Python output.
# ARTIFICIAL NEURAL NETWORK USING BACKPROPAGATION
# 
# DATASET
# ----------------------------------------------------------------------
# Input: [0. 0.] Target: 0
# Input: [0. 1.] Target: 1
# Input: [1. 0.] Target: 1
# Input: [1. 1.] Target: 0
# 
# NETWORK ARCHITECTURE
# ----------------------------------------------------------------------
# Input Layer  : 2 neurons
# Hidden Layer : 4 neurons
# Output Layer : 1 neuron
# Activation   : Sigmoid
# Learning Rate: 0.5
# Epochs       : 10000
# Epoch: 1000 Loss: 0.005865
# Epoch: 2000 Loss: 0.001788
# Epoch: 3000 Loss: 0.001019
# Epoch: 4000 Loss: 0.000705
# Epoch: 5000 Loss: 0.000536
# Epoch: 6000 Loss: 0.000431
# Epoch: 7000 Loss: 0.00036
# Epoch: 8000 Loss: 0.000309
# Epoch: 9000 Loss: 0.00027
# Epoch: 10000 Loss: 0.000239
# 
# FINAL RESULTS
# ----------------------------------------------------------------------
# Input	Target	Output	Prediction
# [0. 0.] 	 0 	 0.0095 	 0
# [0. 1.] 	 1 	 0.9839 	 1
# [1. 0.] 	 1 	 0.9849 	 1
# [1. 1.] 	 0 	 0.0195 	 0
# 
# Accuracy: 100.0 %
# 
# CONFUSION MATRIX
# ----------------------------------------------------------------------
# [[2 0]
#  [0 2]]
# 
# CLASSIFICATION REPORT
# ----------------------------------------------------------------------
#               precision    recall  f1-score   support
# 
#          0.0       1.00      1.00      1.00         2
#          1.0       1.00      1.00      1.00         2
# 
#     accuracy                           1.00         4
#    macro avg       1.00      1.00      1.00         4
# weighted avg       1.00      1.00      1.00         4
# 
# 
# TRAINED WEIGHTS
# ----------------------------------------------------------------------
# Input -> Hidden weights:
# [[-0.40202049  4.60779894  6.37090183  3.82832237]
#  [-1.20298037 -6.26244632 -4.77443726  3.84663039]]
# 
# Hidden -> Output weights:
# [[ 1.5059499 ]
#  [ 8.92519656]
#  [-9.08595643]
#  [ 4.44147084]]
# 
# 
# ANN BACKPROPAGATION DEMONSTRATION COMPLETED
