"""Candidate-Elimination Algorithm.

Extracted from FOML.ipynb.
Run this file with Python to reproduce the console output and plots.
"""

import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# DATASET
# ============================================================

data = pd.DataFrame([
    ['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change', 'No'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change', 'Yes']
], columns=['Sky', 'AirTemp', 'Humidity', 'Wind', 'Water', 'Forecast', 'EnjoySport'])

print('=' * 70)
print('CANDIDATE-ELIMINATION ALGORITHM')
print('=' * 70)

print('\nTRAINING DATA')
print('-' * 70)
print(data.to_string(index=False))

# ============================================================
# PREPARE DATA
# ============================================================

attributes = list(data.columns[:-1])
examples = data.iloc[:, :-1].values.tolist()
labels = data.iloc[:, -1].tolist()
n = len(attributes)

# Possible values of every attribute
domains = [list(data[c].unique()) for c in attributes]

# ============================================================
# INITIAL BOUNDARIES
# ============================================================

S = ['0'] * n
G = [['?'] * n]

# Store graph information
example_numbers = []
s_values = []
g_values = []

# ============================================================
# FUNCTIONS
# ============================================================

def covers(h, x):
    for a, b in zip(h, x):
        if a == '0':
            return False
        if a != '?' and a != b:
            return False
    return True


def more_general_or_equal(h1, h2):
    for a, b in zip(h1, h2):
        if a == '?':
            continue
        if a == '0':
            if b != '0':
                return False
        elif b == '?':
            return False
        elif a != b:
            return False
    return True

# ============================================================
# ALGORITHM
# ============================================================

print('\n' + '=' * 70)
print('CANDIDATE-ELIMINATION PROCESS')
print('=' * 70)

for i in range(len(examples)):

    x = examples[i]
    target = labels[i]

    print('\nExample', i + 1, ':', x, '->', target)

    # --------------------------------------------------------
    # POSITIVE EXAMPLE
    # --------------------------------------------------------

    if target == 'Yes':

        # Remove G hypotheses that do not cover positive example
        G = [g for g in G if covers(g, x)]

        # Generalize S
        for j in range(n):
            if S[j] == '0':
                S[j] = x[j]
            elif S[j] != x[j]:
                S[j] = '?'

        # Keep G hypotheses that are more general than S
        G = [g for g in G if more_general_or_equal(g, S)]

    # --------------------------------------------------------
    # NEGATIVE EXAMPLE
    # --------------------------------------------------------

    else:

        new_G = []

        for g in G:

            if covers(g, x):

                # Specialize G
                for j in range(n):

                    if g[j] == '?':

                        for value in domains[j]:

                            if value != x[j]:

                                new_g = g.copy()
                                new_g[j] = value

                                if more_general_or_equal(new_g, S):
                                    new_G.append(new_g)

            else:
                new_G.append(g)

        G = new_G

    # Remove duplicate G hypotheses
    unique_G = []

    for g in G:
        if g not in unique_G:
            unique_G.append(g)

    G = unique_G

    # --------------------------------------------------------
    # SAVE VALUES FOR GRAPH
    # --------------------------------------------------------

    example_numbers.append(i + 1)

    # Number of generalized attributes in S
    s_count = S.count('?')
    s_values.append(s_count)

    # Average number of generalized attributes in G
    if len(G) > 0:
        g_count = sum(g.count('?') for g in G) / len(G)
    else:
        g_count = 0

    g_values.append(g_count)

    # --------------------------------------------------------
    # DISPLAY CURRENT BOUNDARIES
    # --------------------------------------------------------

    print('Specific Boundary S:')
    print('<' + ', '.join(S) + '>')

    print('General Boundary G:')

    if len(G) == 0:
        print('Empty')
    else:
        for g in G:
            print('<' + ', '.join(g) + '>')

# ============================================================
# FINAL RESULT
# ============================================================

print('\n' + '=' * 70)
print('FINAL RESULT')
print('=' * 70)

print('\nSpecific Boundary S:')
print('<' + ', '.join(S) + '>')

print('\nGeneral Boundary G:')

if len(G) == 0:
    print('Empty')
else:
    for g in G:
        print('<' + ', '.join(g) + '>')

print('\nVersion Space:')
print('All hypotheses between S and G are consistent with the training data.')

# ============================================================
# GRAPH 1 - BOUNDARY EVOLUTION
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    example_numbers,
    s_values,
    marker='o',
    linewidth=2,
    label='Specific Boundary (S)'
)

plt.plot(
    example_numbers,
    g_values,
    marker='s',
    linewidth=2,
    label='General Boundary (G)'
)

plt.xlabel('Training Example Processed')
plt.ylabel('Number of Generalized Attributes (?)')
plt.title('Candidate-Elimination Boundary Evolution')
plt.xticks(example_numbers)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ============================================================
# GRAPH 2 - VERSION SPACE SIZE
# ============================================================

version_space = []

for i in range(len(example_numbers)):

    # Approximate visualization measure based on
    # number of hypotheses in G at each stage
    if i == len(example_numbers) - 1:
        version_space.append(len(G))
    else:
        version_space.append(max(1, len(G)))

plt.figure(figsize=(10, 5))

plt.bar(
    example_numbers,
    version_space
)

plt.xlabel('Training Example Processed')
plt.ylabel('Number of General Boundary Hypotheses')
plt.title('Candidate-Elimination Version Space Progress')
plt.xticks(example_numbers)
plt.grid(axis='y')
plt.tight_layout()
plt.show()

print('\n' + '=' * 70)
print('PROGRAM COMPLETED SUCCESSFULLY')
print('=' * 70)

# ------------------------------------------------------------------
# SAVED CONSOLE OUTPUT (from the original notebook)
# ------------------------------------------------------------------
# This section is a reference; it is not executed as Python output.
# CANDIDATE-ELIMINATION ALGORITHM
# 
# TRAINING DATA
# ----------------------------------------------------------------------
#   Sky AirTemp Humidity   Wind Water Forecast EnjoySport
# Sunny    Warm   Normal Strong  Warm     Same        Yes
# Sunny    Warm     High Strong  Warm     Same        Yes
# Rainy    Cold     High Strong  Warm   Change         No
# Sunny    Warm     High Strong  Cool   Change        Yes
# 
# CANDIDATE-ELIMINATION PROCESS
# 
# Example 1 : ['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same'] -> Yes
# Specific Boundary S:
# <Sunny, Warm, Normal, Strong, Warm, Same>
# General Boundary G:
# <?, ?, ?, ?, ?, ?>
# 
# Example 2 : ['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same'] -> Yes
# Specific Boundary S:
# <Sunny, Warm, ?, Strong, Warm, Same>
# General Boundary G:
# <?, ?, ?, ?, ?, ?>
# 
# Example 3 : ['Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change'] -> No
# Specific Boundary S:
# <Sunny, Warm, ?, Strong, Warm, Same>
# General Boundary G:
# <Sunny, ?, ?, ?, ?, ?>
# <?, Warm, ?, ?, ?, ?>
# <?, ?, ?, ?, ?, Same>
# 
# Example 4 : ['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change'] -> Yes
# Specific Boundary S:
# <Sunny, Warm, ?, Strong, ?, ?>
# General Boundary G:
# <Sunny, ?, ?, ?, ?, ?>
# <?, Warm, ?, ?, ?, ?>
# 
# FINAL RESULT
# 
# Specific Boundary S:
# <Sunny, Warm, ?, Strong, ?, ?>
# 
# General Boundary G:
# <Sunny, ?, ?, ?, ?, ?>
# <?, Warm, ?, ?, ?, ?>
# 
# Version Space:
# All hypotheses between S and G are consistent with the training data.
