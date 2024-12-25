import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, LeakyReLU, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
import seaborn as sns

# Load and preprocess data
print("Loading and preprocessing data...")
data_path = "datasetIRB2400.csv"
data = pd.read_csv(data_path)

# Split features and targets
X = data.drop(columns=["q1_out","q2_out","q3_out","q4_out","q5_out", "q6_out"])
y = data[["q1_out","q2_out","q3_out","q4_out","q5_out", "q6_out"]]

# Split data with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Training set shape:", X_train_scaled.shape)
print("Test set shape:", X_test_scaled.shape)

# Define the residual block
def residual_block(x, units, dropout_rate=0.3):
    shortcut = x
    
    x = Dense(units, kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(dropout_rate)(x)
    
    x = Dense(units, kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    
    # Add shortcut if dimensions match
    if K.int_shape(shortcut)[-1] == units:
        x = Add()([x, shortcut])
    else:
        shortcut = Dense(units)(shortcut)
        x = Add()([x, shortcut])
    
    x = LeakyReLU(alpha=0.1)(x)
    return x

# Build improved model with residual connections
def build_model(input_dim, output_dim):
    inputs = Input(shape=(input_dim,))
    
    # Initial layers
    x = Dense(256, kernel_regularizer=l2(1e-4))(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.2)(x)
    
    # Residual blocks with increasing complexity
    x = residual_block(x, 512, dropout_rate=0.3)
    x = residual_block(x, 512, dropout_rate=0.3)
    x = residual_block(x, 256, dropout_rate=0.2)
    x = residual_block(x, 256, dropout_rate=0.2)
    
    # Output layer
    outputs = Dense(output_dim, activation='linear', kernel_regularizer=l2(1e-4))(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

# Custom loss function combining MSE with angle periodicity
def periodic_mse_loss(y_true, y_pred):
    # Standard MSE
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    
    # Periodic component for angles (assuming angles are in radians)
    angle_diff = tf.math.mod(tf.abs(y_true - y_pred), 2 * np.pi)
    periodic_loss = tf.reduce_mean(tf.minimum(angle_diff, 2 * np.pi - angle_diff))
    
    return mse + 0.1 * periodic_loss

# Initialize model
print("Building model...")
from tensorflow.keras import backend as K
model = build_model(X_train.shape[1], y_train.shape[1])
model.summary()

# Compile model
model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss=periodic_mse_loss,
    metrics=['mse', tf.keras.metrics.RootMeanSquaredError(name='rmse')]
)

# Callbacks
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    ),
    ModelCheckpoint(
        'best_model.keras',  # Changed from .h5 to .keras
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
]

# Train model
print("Training model...")
history = model.fit(
    X_train_scaled,
    y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

# Evaluate model
print("\nEvaluating model...")
test_results = model.evaluate(X_test_scaled, y_test, verbose=1)
print(f"\nTest Loss: {test_results[0]:.4f}")
print(f"Test MSE: {test_results[1]:.4f}")
print(f"Test RMSE: {test_results[2]:.4f}")

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['rmse'], label='Training RMSE')
plt.plot(history.history['val_rmse'], label='Validation RMSE')
plt.title('Model RMSE')
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.legend()

plt.tight_layout()
plt.show()

# Analyze predictions
y_pred = model.predict(X_test_scaled)

# Calculate per-joint errors
joint_errors = np.abs(y_test.values - y_pred)
joint_names = ['q1', 'q2', 'q3', 'q4', 'q5', 'q6']

# Plot error distribution
plt.figure(figsize=(15, 5))
plt.boxplot([joint_errors[:, i] for i in range(6)], labels=joint_names)
plt.title('Joint Prediction Error Distribution')
plt.ylabel('Absolute Error (radians)')
plt.grid(True)
plt.show()

# Print error statistics
print("\nPer-joint error statistics (radians):")
for i, joint in enumerate(joint_names):
    print(f"\n{joint}:")
    print(f"Mean error: {np.mean(joint_errors[:, i]):.4f}")
    print(f"Median error: {np.median(joint_errors[:, i]):.4f}")
    print(f"Max error: {np.max(joint_errors[:, i]):.4f}")

# Save model and scaler
print("\nSaving model and scaler...")
model.save('improved_kinematics_model.keras')  # Changed from .h5 to .keras
import joblib
joblib.dump(scaler, 'scaler.save')

# Function for making predictions
def predict_joint_angles(input_data):
    """Make predictions using the trained model"""
    if input_data.ndim == 1:
        input_data = input_data.reshape(1, -1)
    
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    return prediction

# Example prediction
print("\nExample prediction:")
sample_input = X_test.iloc[0:1]
prediction = predict_joint_angles(sample_input)
actual = y_test.iloc[0:1]

print("\nInput:")
print(sample_input)
print("\nPredicted joint angles:")
print(prediction)
print("\nActual joint angles:")
print(actual)
print("\nAbsolute errors:")
print(np.abs(prediction - actual.values))