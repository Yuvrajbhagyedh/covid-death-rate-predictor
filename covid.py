import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv(r'C:\Users\HP\Downloads\archive\worldometer_data.csv')

# Clean data - drop rows with missing critical values
data = data.dropna(subset=['TotalCases', 'TotalDeaths', 'Population', 'TotalTests'])

# Create target: Death Rate (High if >3%, Low if ≤3%)
data['DeathRate'] = (data['TotalDeaths'] / data['TotalCases']) * 100
data['HighDeathRate'] = (data['DeathRate'] > 3).astype(int)  # 1=High, 0=Low

# Check it
print("Death Rate Statistics:")
print(data['DeathRate'].describe())
print("\nHigh vs Low Death Rate:")
print(data['HighDeathRate'].value_counts())

# Show some examples
print("\nSample data:")
print(data[['Country/Region', 'TotalCases', 'TotalDeaths', 'DeathRate', 'HighDeathRate']].head(10))

# Select features for prediction
features = ['Population', 'TotalTests', 'Tests/1M pop', 'Tot Cases/1M pop']

# Prepare X (features) and y (target)
X = data[features]
y = data['HighDeathRate']

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the model
model = DecisionTreeClassifier(random_state=42, max_depth=5)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Check accuracy
accuracy = accuracy_score(y_test, predictions)
print("\n" + "="*50)
print("MODEL RESULTS:")
print("="*50)
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"\nTested on {len(y_test)} countries")
print(f"Correctly predicted: {sum(predictions == y_test)} countries")
print(f"Incorrectly predicted: {sum(predictions != y_test)} countries")

plt.figure(figsize=(20,10))
tree.plot_tree(model, filled=True, feature_names=features)
plt.show()