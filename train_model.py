import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import os
import sys

# 2. Load the dataset using pandas
dataset_name = "financial_dataset_project.csv"
dataset_path = None


# First check the current directory
if os.path.exists(dataset_name):
    dataset_path = os.path.abspath(dataset_name)
else:
    # Search for the dataset inside the project workspace
    project_dir = os.path.dirname(os.path.abspath(__file__))
    for root, dirs, files in os.walk(project_dir):
        if dataset_name in files:
            dataset_path = os.path.join(root, dataset_name)
            break
    
    # As a fallback, try searching one directory up in case it was placed outside
    if not dataset_path:
        parent_dir = os.path.dirname(project_dir)
        for root, dirs, files in os.walk(parent_dir):
            if dataset_name in files:
                dataset_path = os.path.join(root, dataset_name)
                break

if not dataset_path:
    print(f"Error: The dataset file '{dataset_name}' was not found.")
    print(f"Please ensure it is placed anywhere inside your project folder: {os.path.dirname(os.path.abspath(__file__))}")
    sys.exit(1)

print(f"Dataset found at: {dataset_path}")
print("Loading dataset...")
df = pd.read_csv(dataset_path)
print("Dataset loaded successfully.")

# 3. Print basic dataset information
print("\n--- Dataset Head ---")
print(df.head())

print("\n--- Dataset Shape ---")
print(df.shape)

print("\n--- Dataset Columns ---")
print(df.columns.tolist())

print("\n--- Missing Values ---")
print(df.isnull().sum())

# 4. Convert the "recommendations" column into numeric values using LabelEncoder
label_encoder = LabelEncoder()
df['recommendations'] = label_encoder.fit_transform(df['recommendations'])

# 5. Separate features and target variable
# If there are other categorical columns (like 'risk_level'), pandas get_dummies is used
# to encode them so the Linear Regression model doesn't throw an error.
X_temp = df.drop(columns=["recommendations"])
X = pd.get_dummies(X_temp, drop_first=True) 

y = df["recommendations"]

# 6. Split the dataset: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Train a Linear Regression model using sklearn
print("\nTraining Linear Regression model...")
model = LinearRegression()
model.fit(X_train, y_train)

# 8. Evaluate the model
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

# 9. Print model performance results
print("\n--- Model Performance ---")
print(f"R2 Score: {r2}")
print(f"Mean Squared Error: {mse}")

# 10. Save the trained model using pickle
with open("investment_model.pkl", "wb") as f:
    pickle.dump(model, f)
    
# (Optional but recommended) Save the label encoder to decode predictions later
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

# 11. Print confirmation message
print("\nModel trained and saved successfully.")
