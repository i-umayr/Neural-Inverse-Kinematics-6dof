import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd
import matplotlib.pyplot as plt

# Load the model
model = load_model('best_model.keras')

# Load the scaler
scaler = joblib.load('scaler.save')

# Load your dataset (or use a separate test dataset)
data_path = "datasetIRB2400.csv"  # Change this to your test dataset path
data = pd.read_csv(data_path)

# Split features and targets
X = data.drop(columns=["q1_out", "q2_out", "q3_out", "q4_out", "q5_out", "q6_out"])
y = data[["q1_out", "q2_out", "q3_out", "q4_out", "q5_out", "q6_out"]]

# Split the data into training and test sets (if not already done)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the test data
X_test_scaled = scaler.transform(X_test)

# Evaluate the model
print("\nEvaluating model...")
test_results = model.evaluate(X_test_scaled, y_test, verbose=1)
print(f"\nTest Loss: {test_results[0]:.4f}")
print(f"Test MSE: {test_results[1]:.4f}")
print(f"Test RMSE: {test_results[2]:.4f}")

# Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# Calculate errors
joint_errors = np.abs(y_test.values - y_pred)

# Plot error distribution
plt.figure(figsize=(15, 5))
plt.boxplot([joint_errors[:, i] for i in range(6)], labels=['q1', 'q2', 'q3', 'q4', 'q5', 'q6'])
plt.title('Joint Prediction Error Distribution')
plt.ylabel('Absolute Error (radians)')
plt.grid(True)
plt.show()

# Print per-joint error statistics
joint_names = ['q1', 'q2', 'q3', 'q4', 'q5', 'q6']
print("\nPer-joint error statistics (radians):")
for i, joint in enumerate(joint_names):
    print(f"\n{joint}:")
    print(f"Mean error: {np.mean(joint_errors[:, i]):.4f}")
    print(f"Median error: {np.median(joint_errors[:, i]):.4f}")
    print(f"Max error: {np.max(joint_errors[:, i]):.4f}")