"""ID3 Decision Tree Algorithm.

Extracted from FOML.ipynb.
Run this file with Python to reproduce the console output and plots.
"""

import pandas as pd
from math import log2
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# ============================================================
# ID3 DECISION TREE ALGORITHM
# ============================================================

# Dataset

data = pd.DataFrame([
['Sunny','Hot','High','Weak','No'],
['Sunny','Hot','High','Strong','No'],
['Overcast','Hot','High','Weak','Yes'],
['Rain','Mild','High','Weak','Yes'],
['Rain','Cool','Normal','Weak','Yes'],
['Rain','Cool','Normal','Strong','No'],
['Overcast','Cool','Normal','Strong','Yes'],
['Sunny','Mild','High','Weak','No'],
['Sunny','Cool','Normal','Weak','Yes'],
['Rain','Mild','Normal','Weak','Yes'],
['Sunny','Mild','Normal','Strong','Yes'],
['Overcast','Mild','High','Strong','Yes'],
['Overcast','Hot','Normal','Weak','Yes'],
['Rain','Mild','High','Strong','No']
], columns=['Outlook','Temperature','Humidity','Wind','PlayTennis'])

print('='*70)
print('ID3 DECISION TREE ALGORITHM')
print('='*70)

print('\nTRAINING DATA')
print('-'*70)
print(data.to_string(index=False))

# ============================================================
# ENTROPY FUNCTION
# ============================================================

def entropy(values):
    counts = values.value_counts()
    total = len(values)
    result = 0
    for count in counts:
        probability = count / total
        result -= probability * log2(probability)
    return result

# ============================================================
# INFORMATION GAIN
# ============================================================

def information_gain(data, attribute, target):
    total_entropy = entropy(data[target])
    weighted_entropy = 0
    for value in data[attribute].unique():
        subset = data[data[attribute] == value]
        weighted_entropy += (len(subset) / len(data)) * entropy(subset[target])
    return total_entropy - weighted_entropy

# ============================================================
# CALCULATE ENTROPY
# ============================================================

target = 'PlayTennis'
features = ['Outlook','Temperature','Humidity','Wind']

print('\nINITIAL ENTROPY')
print('-'*70)
print('Entropy of PlayTennis =', round(entropy(data[target]), 4))

# ============================================================
# CALCULATE INFORMATION GAIN
# ============================================================

print('\nINFORMATION GAIN')
print('-'*70)

for feature in features:
    gain = information_gain(data, feature, target)
    print(feature, '=', round(gain, 4))

# Find best root attribute
best_feature = max(
    features,
    key=lambda feature: information_gain(data, feature, target)
)

print('\nBest Root Attribute:', best_feature)

# ============================================================
# BUILD ID3 TREE
# ============================================================

X = data[features]
y = data[target]

# Encode categorical data
X_encoded = pd.get_dummies(X)

# ID3 uses entropy criterion
model = DecisionTreeClassifier(
    criterion='entropy',
    random_state=42
)

model.fit(X_encoded, y)

# ============================================================
# DISPLAY TREE STRUCTURE
# ============================================================

print('\nDECISION TREE CREATED SUCCESSFULLY')
print('-'*70)
print('Root Node:', best_feature)

# ============================================================
# CLASSIFY NEW SAMPLE
# ============================================================

new_sample = pd.DataFrame([
    ['Sunny','Cool','High','Strong']
], columns=features)

# Convert new sample using same columns as training data
new_sample_encoded = pd.get_dummies(new_sample)
new_sample_encoded = new_sample_encoded.reindex(
    columns=X_encoded.columns,
    fill_value=0
)

prediction = model.predict(new_sample_encoded)[0]

print('\nNEW SAMPLE')
print('-'*70)
print('Outlook     : Sunny')
print('Temperature : Cool')
print('Humidity    : High')
print('Wind        : Strong')

print('\nCLASSIFICATION RESULT')
print('-'*70)
print('Prediction:', prediction)

if prediction == 'Yes':
    print('The decision tree predicts: PLAY TENNIS')
else:
    print('The decision tree predicts: DO NOT PLAY TENNIS')

# ============================================================
# DISPLAY TREE GRAPH
# ============================================================

plt.figure(figsize=(18,10))

plot_tree(
    model,
    feature_names=X_encoded.columns,
    class_names=model.classes_,
    filled=True,
    rounded=True,
    fontsize=9
)

plt.title('ID3 Decision Tree - Play Tennis Dataset')
plt.tight_layout()
plt.show()

# ============================================================
# INFORMATION GAIN GRAPH
# ============================================================

gains = []

for feature in features:
    gains.append(information_gain(data, feature, target))

plt.figure(figsize=(9,5))

plt.bar(features, gains)

plt.xlabel('Attributes')
plt.ylabel('Information Gain')
plt.title('Information Gain of Attributes - ID3')
plt.grid(axis='y')
plt.tight_layout()
plt.show()

print('\n'+'='*70)
print('ID3 DEMONSTRATION COMPLETED SUCCESSFULLY')
print('='*70)

# ------------------------------------------------------------------
# SAVED CONSOLE OUTPUT (from the original notebook)
# ------------------------------------------------------------------
# This section is a reference; it is not executed as Python output.
# ID3 DECISION TREE ALGORITHM
# 
# TRAINING DATA
# ----------------------------------------------------------------------
#  Outlook Temperature Humidity   Wind PlayTennis
#    Sunny         Hot     High   Weak         No
#    Sunny         Hot     High Strong         No
# Overcast         Hot     High   Weak        Yes
#     Rain        Mild     High   Weak        Yes
#     Rain        Cool   Normal   Weak        Yes
#     Rain        Cool   Normal Strong         No
# Overcast        Cool   Normal Strong        Yes
#    Sunny        Mild     High   Weak         No
#    Sunny        Cool   Normal   Weak        Yes
#     Rain        Mild   Normal   Weak        Yes
#    Sunny        Mild   Normal Strong        Yes
# Overcast        Mild     High Strong        Yes
# Overcast         Hot   Normal   Weak        Yes
#     Rain        Mild     High Strong         No
# 
# INITIAL ENTROPY
# ----------------------------------------------------------------------
# Entropy of PlayTennis = 0.9403
# 
# INFORMATION GAIN
# ----------------------------------------------------------------------
# Outlook = 0.2467
# Temperature = 0.0292
# Humidity = 0.1518
# Wind = 0.0481
# 
# Best Root Attribute: Outlook
# 
# DECISION TREE CREATED SUCCESSFULLY
# ----------------------------------------------------------------------
# Root Node: Outlook
# 
# NEW SAMPLE
# ----------------------------------------------------------------------
# Outlook     : Sunny
# Temperature : Cool
# Humidity    : High
# Wind        : Strong
# 
# CLASSIFICATION RESULT
# ----------------------------------------------------------------------
# Prediction: No
# The decision tree predicts: DO NOT PLAY TENNIS
