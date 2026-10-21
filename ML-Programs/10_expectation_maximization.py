"""Expectation-Maximization (EM) Algorithm.

Extracted from FOML.ipynb.
Run this file with Python to reproduce the console output and plots.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

print('='*70)
print('EXPECTATION-MAXIMIZATION (EM) ALGORITHM')
print('='*70)

# Dataset: Two-dimensional points belonging to two hidden groups
X = np.array([
[1.0,1.5],[1.2,1.8],[1.5,1.2],[1.8,1.6],[2.0,2.0],
[2.2,1.5],[2.5,2.2],[2.8,1.8],[3.0,2.5],[3.2,2.0],
[7.0,7.5],[7.2,8.0],[7.5,7.2],[7.8,7.8],[8.0,8.5],
[8.2,7.5],[8.5,8.2],[8.8,7.8],[9.0,8.5],[9.2,9.0]
])

print('\nDATASET')
print('-'*70)
print('Total samples:', len(X))
print('Features: X and Y')
print('\nFirst 10 samples:')
for i in range(10):
    print(X[i])

# Number of clusters
n_clusters = 2

# Create Gaussian Mixture Model using EM
model = GaussianMixture(n_components=n_clusters, covariance_type='full', random_state=42, max_iter=100)

print('\nEM MODEL')
print('-'*70)
print('Number of clusters:', n_clusters)
print('Distribution: Gaussian')
print('Maximum iterations:', 100)

# Fit the model
model.fit(X)

# Predict cluster assignments
labels = model.predict(X)

# Get probability of each sample belonging to each cluster
probabilities = model.predict_proba(X)

print('\nEM ALGORITHM RESULTS')
print('-'*70)
print('Converged:', model.converged_)
print('Number of iterations:', model.n_iter_)
print('Final log-likelihood:', round(model.lower_bound_, 6))

print('\nESTIMATED MEANS')
print('-'*70)
for i, mean in enumerate(model.means_):
    print('Cluster', i + 1, ':', np.round(mean, 4))

print('\nESTIMATED COVARIANCE MATRICES')
print('-'*70)
for i, covariance in enumerate(model.covariances_):
    print('Cluster', i + 1, ':')
    print(np.round(covariance, 4))

print('\nCLUSTER ASSIGNMENTS')
print('-'*70)
print('Sample\tX\tY\tCluster')
for i in range(len(X)):
    print(i + 1, '\t', X[i,0], '\t', X[i,1], '\t', labels[i] + 1)

# Display membership probabilities
print('\nMEMBERSHIP PROBABILITIES - FIRST 10 SAMPLES')
print('-'*70)
for i in range(10):
    print('Sample', i + 1, ':', np.round(probabilities[i], 4))

# Predict a new sample
new_sample = np.array([[2.0,1.7]])
new_cluster = model.predict(new_sample)[0]
new_probability = model.predict_proba(new_sample)[0]

print('\nNEW SAMPLE CLASSIFICATION')
print('-'*70)
print('New sample:', new_sample[0])
print('Predicted cluster:', new_cluster + 1)
print('Cluster probabilities:', np.round(new_probability, 4))

# Plot clusters
plt.figure(figsize=(9,6))
for cluster in range(n_clusters):
    points = X[labels == cluster]
    plt.scatter(points[:,0], points[:,1], s=70, label='Cluster ' + str(cluster + 1))

plt.scatter(model.means_[:,0], model.means_[:,1], marker='X', s=200, label='Cluster Centers')
plt.scatter(new_sample[:,0], new_sample[:,1], marker='*', s=250, label='New Sample')
plt.xlabel('Feature X')
plt.ylabel('Feature Y')
plt.title('Expectation-Maximization Clustering using GMM')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print('\n'+'='*70)
print('EXPECTATION-MAXIMIZATION IMPLEMENTATION COMPLETED SUCCESSFULLY')
print('='*70)

# ------------------------------------------------------------------
# SAVED CONSOLE OUTPUT (from the original notebook)
# ------------------------------------------------------------------
# This section is a reference; it is not executed as Python output.
# EXPECTATION-MAXIMIZATION (EM) ALGORITHM
# 
# DATASET
# ----------------------------------------------------------------------
# Total samples: 20
# Features: X and Y
# 
# First 10 samples:
# [1.  1.5]
# [1.2 1.8]
# [1.5 1.2]
# [1.8 1.6]
# [2. 2.]
# [2.2 1.5]
# [2.5 2.2]
# [2.8 1.8]
# [3.  2.5]
# [3.2 2. ]
# 
# EM MODEL
# ----------------------------------------------------------------------
# Number of clusters: 2
# Distribution: Gaussian
# Maximum iterations: 100
# 
# EM ALGORITHM RESULTS
# ----------------------------------------------------------------------
# Converged: True
# Number of iterations: 2
# Final log-likelihood: -2.086635
# 
# ESTIMATED MEANS
# ----------------------------------------------------------------------
# Cluster 1 : [2.12 1.81]
# Cluster 2 : [8.12 8.  ]
# 
# ESTIMATED COVARIANCE MATRICES
# ----------------------------------------------------------------------
# Cluster 1 :
# [[0.5156 0.1708]
#  [0.1708 0.1309]]
# Cluster 2 :
# [[0.5156 0.248 ]
#  [0.248  0.276 ]]
# 
# CLUSTER ASSIGNMENTS
# ----------------------------------------------------------------------
# Sample	X	Y	Cluster
# 1 	 1.0 	 1.5 	 1
# 2 	 1.2 	 1.8 	 1
# 3 	 1.5 	 1.2 	 1
# 4 	 1.8 	 1.6 	 1
# 5 	 2.0 	 2.0 	 1
# 6 	 2.2 	 1.5 	 1
# 7 	 2.5 	 2.2 	 1
# 8 	 2.8 	 1.8 	 1
# 9 	 3.0 	 2.5 	 1
# 10 	 3.2 	 2.0 	 1
# 11 	 7.0 	 7.5 	 2
# 12 	 7.2 	 8.0 	 2
# 13 	 7.5 	 7.2 	 2
# 14 	 7.8 	 7.8 	 2
# 15 	 8.0 	 8.5 	 2
# 16 	 8.2 	 7.5 	 2
# 17 	 8.5 	 8.2 	 2
# 18 	 8.8 	 7.8 	 2
# 19 	 9.0 	 8.5 	 2
# 20 	 9.2 	 9.0 	 2
# 
# MEMBERSHIP PROBABILITIES - FIRST 10 SAMPLES
# ----------------------------------------------------------------------
# Sample 1 : [1. 0.]
# Sample 2 : [1. 0.]
# Sample 3 : [1. 0.]
# Sample 4 : [1. 0.]
# Sample 5 : [1. 0.]
# Sample 6 : [1. 0.]
# Sample 7 : [1. 0.]
# Sample 8 : [1. 0.]
# Sample 9 : [1. 0.]
# Sample 10 : [1. 0.]
# 
# NEW SAMPLE CLASSIFICATION
# ----------------------------------------------------------------------
# New sample: [2.  1.7]
# Predicted cluster: 1
# Cluster probabilities: [1. 0.]
