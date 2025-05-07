import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

# Load the dataset (replace with your dataset path)
data = pd.read_csv('creditcard.csv')
# Use only a subset of the data for quicker experimentation
data_subset = data.sample(frac=0.1, random_state=42)  # Use 10% of the data
X = data_subset.drop('Class', axis=1)
y = data_subset['Class']


# Preprocess data (this step depends on your dataset)
X = data.drop('Class', axis=1)  # Features
y = data['Class']  # Target variable
print("1")

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("2")
# Standardize the data (important for some models)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print("3")
# Initialize and train the model
model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X_train, y_train)
print('4')
# Save the model to a file
with open('card_fraud_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model saved as card_fraud_model.pkl")
