import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Create dataset
data = {
    'Income': [25000, 30000, 45000, 50000, 60000,
               20000, 70000, 80000, 35000, 90000],
    'Loan': [15000, 20000, 10000, 25000, 20000,
             18000, 15000, 30000, 22000, 25000],
    'Payment_History': [2, 3, 7, 6, 8, 1, 9, 9, 4, 10],
    'Credit_Score': ['Poor', 'Poor', 'Good', 'Good', 'Good',
                     'Poor', 'Good', 'Good', 'Poor', 'Good']
}

df = pd.DataFrame(data)

# Separate input and output
X = df[['Income', 'Loan', 'Payment_History']]
y = df['Credit_Score']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train the model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict credit scores
y_pred = model.predict(X_test)

# Display accuracy
print("Predicted Credit Scores:", list(y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred) * 100, "%")

# Visualize using Matplotlib
plt.scatter(df['Income'], df['Loan'],
            c=df['Credit_Score'].map({'Poor': 0, 'Good': 1}),
            cmap='coolwarm', s=100)

plt.xlabel("Income")
plt.ylabel("Loan Amount")
plt.title("Credit Score Classification")
plt.colorbar(label="0 = Poor, 1 = Good")
plt.grid(True)
plt.show()

