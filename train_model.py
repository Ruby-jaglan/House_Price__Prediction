import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Get the current project directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to dataset
dataset_path = os.path.join(BASE_DIR, "dataset", "Housing.csv")

# Check if dataset exists
if not os.path.exists(dataset_path):
    print("Error: Housing.csv not found!")
    print("Expected location:", dataset_path)
    exit()

# Load dataset
data = pd.read_csv(dataset_path)

# Convert categorical columns into numeric values
data = pd.get_dummies(data, drop_first=True)

# Features and target
X = data.drop("price", axis=1)
y = data["price"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Create models folder if it doesn't exist
models_dir = os.path.join(BASE_DIR, "models")
os.makedirs(models_dir, exist_ok=True)

# Save model and feature names
joblib.dump(model, os.path.join(models_dir, "house_price_model.pkl"))
joblib.dump(X.columns.tolist(), os.path.join(models_dir, "features.pkl"))

print("✅ Model trained successfully!")