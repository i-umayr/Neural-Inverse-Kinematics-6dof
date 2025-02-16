import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Layer
import matplotlib.pyplot as plt
import joblib

# Define custom layer without decorator
class SequenceGenerationLayer(Layer):
    def __init__(self, sequence_length, **kwargs):
        super().__init__(**kwargs)
        self.sequence_length = sequence_length

    def call(self, inputs):
        expanded = tf.expand_dims(inputs, axis=1)
        return tf.repeat(expanded, repeats=self.sequence_length, axis=1)
    
    def get_config(self):
        config = super().get_config()
        config.update({"sequence_length": self.sequence_length})
        return config

# Custom loss function
def joint_angle_loss(y_true, y_pred):
    """Custom loss function for joint angles considering periodicity"""
    # Regular MSE
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    
    # Periodic component
    angle_diff = tf.math.mod(tf.abs(y_true - y_pred), 2 * np.pi)
    periodic_loss = tf.reduce_mean(tf.minimum(angle_diff, 2 * np.pi - angle_diff))
    
    return 0.7 * mse + 0.3 * periodic_loss

try:
    # Load the saved model with custom objects
    print("Loading model and scalers...")
    custom_objects = {
        'SequenceGenerationLayer': SequenceGenerationLayer,
        'joint_angle_loss': joint_angle_loss
    }
    
    model = tf.keras.models.load_model('advanced_kinematics_model.keras', 
                                     custom_objects=custom_objects)

    # Load the scalers
    standard_scaler = joblib.load('standard_scaler.save')
    minmax_scaler = joblib.load('minmax_scaler.save')

    # Load and preprocess the dataset
    print("Loading and preprocessing data...")
    data = pd.read_csv("datasetIRB2400.csv")

    # Split features and targets
    X = data.drop(columns=["q1_out","q2_out","q3_out","q4_out","q5_out", "q6_out"])
    y = data[["q1_out","q2_out","q3_out","q4_out","q5_out", "q6_out"]]

    # Split data using the same random state as training
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the test data
    X_test_scaled = standard_scaler.transform(X_test)
    y_test_scaled = minmax_scaler.transform(y_test)

    # Evaluate model
    print("\nEvaluating model on test set...")
    test_results = model.evaluate(X_test_scaled, y_test_scaled, verbose=1)
    print(f"\nTest Loss: {test_results[0]:.4f}")
    print(f"Test MSE: {test_results[1]:.4f}")
    print(f"Test RMSE: {test_results[2]:.4f}")

    # Make predictions
    print("\nMaking predictions...")
    y_pred_scaled = model.predict(X_test_scaled)
    y_pred = minmax_scaler.inverse_transform(y_pred_scaled)

    # Calculate errors
    joint_errors = np.abs(y_test.values - y_pred)

    # Plot error distribution
    plt.figure(figsize=(15, 5))
    plt.boxplot([joint_errors[:, i] for i in range(6)], 
                labels=[f'Joint {i+1}' for i in range(6)])
    plt.title('Joint Prediction Error Distribution')
    plt.ylabel('Absolute Error (radians)')
    plt.grid(True)
    plt.show()

    # Print detailed error statistics
    print("\nPer-joint error statistics (radians):")
    for i in range(6):
        print(f"\nJoint {i+1}:")
        errors = joint_errors[:, i]
        print(f"Mean error: {np.mean(errors):.6f}")
        print(f"Median error: {np.median(errors):.6f}")
        print(f"Max error: {np.max(errors):.6f}")
        print(f"95th percentile error: {np.percentile(errors, 95):.6f}")
        print(f"Standard deviation: {np.std(errors):.6f}")

    # Sample predictions
    print("\nSample predictions (first 5 test cases):")
    print("\nActual joint angles (radians):")
    print(y_test.iloc[:5])
    print("\nPredicted joint angles (radians):")
    print(pd.DataFrame(y_pred[:5], columns=y_test.columns))
    print("\nAbsolute errors (radians):")
    print(pd.DataFrame(joint_errors[:5], columns=y_test.columns))

    # Performance testing
    print("\nPerformance analysis...")
    import time

    # Warm-up run
    _ = model.predict(X_test_scaled[:1])

    # Timing test
    num_samples = 1000
    times = []
    for _ in range(10):  # 10 batches of timing tests
        start_time = time.time()
        _ = model.predict(X_test_scaled[:num_samples])
        end_time = time.time()
        times.append((end_time - start_time) / num_samples * 1000)  # Convert to milliseconds

    avg_time = np.mean(times)
    std_time = np.std(times)

    print(f"\nAverage prediction time per sample: {avg_time:.3f} ± {std_time:.3f} ms")
    print(f"Predictions per second: {1000/avg_time:.1f}")

except Exception as e:
    print(f"An error occurred: {str(e)}")
    print("\nDetailed error information:")
    import traceback
    traceback.print_exc()