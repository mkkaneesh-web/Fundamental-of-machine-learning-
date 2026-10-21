"""FIND-S Algorithm.

Extracted from FOML.ipynb.
Run this file with Python to reproduce the console output and plots.
"""

import matplotlib.pyplot as plt
h=['0','0','0','0']
history=[]
data=[['Sunny','Warm','Normal','Strong','Yes'],['Sunny','Warm','High','Strong','Yes'],['Rainy','Cold','High','Strong','No'],['Sunny','Warm','High','Weak','Yes'],['Rainy','Warm','Normal','Weak','No']]
print('FIND-S ALGORITHM')
print('='*50)
for r in data:
 x=r[:4]
 y=r[4]
 if y=='Yes':
  if h==['0','0','0','0']:
   h=x.copy()
  else:
   h=[h[i] if h[i]==x[i] else '?' for i in range(4)]
  history.append(h.copy())
  print('Positive:',x,'->',h)
 else:
  print('Negative:',x,'-> ignored')
print('='*50)
print('FINAL HYPOTHESIS:',h)
plt.figure(figsize=(10,6))
plt.plot([1,2,3],[1,1,1],marker='o',label='Weather')
plt.plot([1,2,3],[1,1,1],marker='o',label='Temperature')
plt.plot([1,2,3],[1,2,2],marker='o',label='Humidity')
plt.plot([1,2,3],[1,1,2],marker='o',label='Wind')
plt.xlabel('Positive Example Processed')
plt.ylabel('Hypothesis State')
plt.title('FIND-S Hypothesis Generalization')
plt.xticks([1,2,3],['Example 1','Example 2','Example 4'])
plt.yticks([1,2],['Specific Value','Generalized (?)'])
plt.grid(True)
plt.legend()
plt.show()
plt.figure(figsize=(9,5))
plt.plot([1,2,3],[0,1,2],marker='o',linewidth=2)
plt.xlabel('Positive Example Processed')
plt.ylabel('Number of Generalized Attributes')
plt.title('FIND-S Generalization Progress')
plt.xticks([1,2,3],['Example 1','Example 2','Example 4'])
plt.grid(True)
plt.show()

# ------------------------------------------------------------------
# SAVED CONSOLE OUTPUT (from the original notebook)
# ------------------------------------------------------------------
# This section is a reference; it is not executed as Python output.
# FIND-S ALGORITHM
# Positive: ['Sunny', 'Warm', 'Normal', 'Strong'] -> ['Sunny', 'Warm', 'Normal', 'Strong']
# Positive: ['Sunny', 'Warm', 'High', 'Strong'] -> ['Sunny', 'Warm', '?', 'Strong']
# Negative: ['Rainy', 'Cold', 'High', 'Strong'] -> ignored
# Positive: ['Sunny', 'Warm', 'High', 'Weak'] -> ['Sunny', 'Warm', '?', '?']
# Negative: ['Rainy', 'Warm', 'Normal', 'Weak'] -> ignored
# FINAL HYPOTHESIS: ['Sunny', 'Warm', '?', '?']
