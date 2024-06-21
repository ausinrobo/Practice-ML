import pandas as pd
import streamlit as st

import pandas as pd

# Load the datasets
train_data = pd.read_excel('Task1and2/train.xlsx')
test_data = pd.read_excel('Task1and2/test.xlsx')

# Inspect the datasets
print("Training Data:")
print(train_data.head())
print("\nTest Data:")
print(test_data.head())


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# Separate features and target variable from training data
X_train = train_data.drop('target', axis=1)
y_train = train_data['target']

# Identify categorical columns
categorical_cols = X_train.select_dtypes(include=['object']).columns

# Apply Label Encoding to categorical columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col])
    label_encoders[col] = le

# Apply the same transformation to test data
for col in categorical_cols:
    test_data[col] = label_encoders[col].transform(test_data[col])

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
test_data = imputer.transform(test_data)
# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
test_data = scaler.transform(test_data)

# Split the data into training and validation sets
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Logistic Regression
log_reg = LogisticRegression()
log_reg.fit(X_train_split, y_train_split)
y_pred_log_reg = log_reg.predict(X_val_split)
log_reg_accuracy = accuracy_score(y_val_split, y_pred_log_reg)

# Random Forest
rf_clf = RandomForestClassifier()
rf_clf.fit(X_train_split, y_train_split)
y_pred_rf = rf_clf.predict(X_val_split)
rf_accuracy = accuracy_score(y_val_split, y_pred_rf)

print(f"Logistic Regression Accuracy: {log_reg_accuracy}")
print(f"Random Forest Accuracy: {rf_accuracy}")


st.title("Machine Learning")

# Task 2: Classification Results
st.header("Task 2: Classification Results")

# Load the test predictions
test_predictions = pd.read_csv('Task1and2/test_predictions.csv')
st.write(test_predictions)

st.write("Logistic Regression Accuracy:", log_reg_accuracy)
st.write("Random Forest Accuracy:", rf_accuracy)


if __name__ == '__main__':
    st.run()
