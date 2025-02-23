import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, LSTM, Dropout, BatchNormalization, Input, 
    LeakyReLU, Add, Concatenate, LayerNormalization,
    GaussianNoise, Layer
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint,
    TensorBoard
)
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
from datetime import datetime


class Config:
    # Increased batch size for larger dataset
    BATCH_SIZE = 128
    EPOCHS = 100
    # Adjusted learning rate for larger batch size
    LEARNING_RATE = 3e-4
    LSTM_UNITS = 256
    DENSE_UNITS = [512, 256, 128]
    DROPOUT_RATE = 0.1
    L2_LAMBDA = 1e-6
    NOISE_FACTOR = 0.01
    # Adjusted validation split for larger dataset
    VALIDATION_SPLIT = 0.15
    # Adjusted patience values for longer training
    EARLY_STOPPING_PATIENCE = 25
    LR_REDUCTION_PATIENCE = 10
    LR_REDUCTION_FACTOR = 0.5
    MIN_LR = 1e-7
    SEQUENCE_LENGTH = 4


class SequenceGenerationLayer(Layer):
    def __init__(self, sequence_length, **kwargs):
        super().__init__(**kwargs)
        self.sequence_length = sequence_length

    def call(self, inputs):
        expanded = tf.expand_dims(inputs, axis=1)
        return tf.repeat(expanded, repeats=self.sequence_length, axis=1)

def joint_angle_loss(y_true, y_pred):
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    
    continuous_joints = [0, 2, 4]
    
    periodic_losses = []
    for idx in continuous_joints:
        angle_diff = tf.math.mod(tf.abs(y_true[:, idx] - y_pred[:, idx]), 2 * np.pi)
        periodic_loss = tf.reduce_mean(tf.minimum(angle_diff, 2 * np.pi - angle_diff))
        periodic_losses.append(periodic_loss)
    
    avg_periodic_loss = tf.reduce_mean(periodic_losses)
    
    return 0.7 * mse + 0.3 * avg_periodic_loss

def residual_block(x, units, dropout_rate=0.1):
    shortcut = x
    
    x = Dense(units, kernel_regularizer=l2(Config.L2_LAMBDA))(x)
    x = LayerNormalization()(x)
    x = LeakyReLU(negative_slope=0.1)(x)
    x = Dropout(dropout_rate)(x)
    
    x = Dense(units, kernel_regularizer=l2(Config.L2_LAMBDA))(x)
    x = LayerNormalization()(x)
    
    if shortcut.shape[-1] != units:
        shortcut = Dense(units)(shortcut)
    
    x = Add()([x, shortcut])
    x = LeakyReLU(negative_slope=0.1)(x)
    return x

def build_advanced_model(input_dim, output_dim):
    main_input = Input(shape=(input_dim,))
    noisy_input = GaussianNoise(Config.NOISE_FACTOR)(main_input)
    
    x = Dense(Config.DENSE_UNITS[0], kernel_regularizer=l2(Config.L2_LAMBDA))(noisy_input)
    x = LayerNormalization()(x)
    x = LeakyReLU(negative_slope=0.1)(x)
    x = Dropout(Config.DROPOUT_RATE)(x)
    
    x_lstm = SequenceGenerationLayer(Config.SEQUENCE_LENGTH)(x)
    x_lstm = LSTM(Config.LSTM_UNITS)(x_lstm)
    
    x_res = x
    for units in Config.DENSE_UNITS:
        x_res = residual_block(x_res, units, Config.DROPOUT_RATE)
    
    combined = Concatenate()([x_lstm, x_res])
    
    x = Dense(256, kernel_regularizer=l2(Config.L2_LAMBDA))(combined)
    x = LayerNormalization()(x)
    x = LeakyReLU(negative_slope=0.1)(x)
    x = Dropout(Config.DROPOUT_RATE)(x)
    
    outputs = Dense(output_dim, activation='linear', 
                   kernel_regularizer=l2(Config.L2_LAMBDA))(x)
    
    return Model(inputs=main_input, outputs=outputs)

def main():
    print("Loading and preprocessing data...")
    # Use chunks to load large dataset efficiently
    data = pd.read_csv("robot_dataset.csv", chunksize=10000)
    data = pd.concat(data, ignore_index=True)
    
    input_features = [f'q{i}_current' for i in range(1, 7)] + \
                    ['x_desired', 'y_desired', 'z_desired', 
                     'roll', 'pitch', 'yaw']
    
    output_features = [f'q{i}_desired' for i in range(1, 7)]
    
    X = data[input_features]
    y = data[output_features]
    
    # Split data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, 
        test_size=Config.VALIDATION_SPLIT, 
        random_state=42,
        shuffle=True
    )
    
    # Scale the data
    input_scaler = StandardScaler(copy=False)
    output_scaler = MinMaxScaler(copy=False)
    
    X_train_scaled = input_scaler.fit_transform(X_train)
    X_val_scaled = input_scaler.transform(X_val)
    
    y_train_scaled = output_scaler.fit_transform(y_train)
    y_val_scaled = output_scaler.transform(y_val)
    
    # Create tf.data.Dataset for training
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (X_train_scaled, y_train_scaled)
    ).shuffle(
        buffer_size=10000
    ).batch(
        Config.BATCH_SIZE
    ).prefetch(
        tf.data.AUTOTUNE
    ).repeat()
    
    # Create validation dataset
    val_dataset = tf.data.Dataset.from_tensor_slices(
        (X_val_scaled, y_val_scaled)
    ).batch(Config.BATCH_SIZE)
    
    print("Building model...")
    model = build_advanced_model(X_train.shape[1], y_train.shape[1])
    model.summary()
    
    # Use mixed precision for faster training
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    
    optimizer = Adam(learning_rate=Config.LEARNING_RATE)
    optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
    
    model.compile(
        optimizer=optimizer,
        loss=joint_angle_loss,
        metrics=['mse', tf.keras.metrics.RootMeanSquaredError(name='rmse')]
    )
    
    log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=Config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=Config.LR_REDUCTION_FACTOR,
            patience=Config.LR_REDUCTION_PATIENCE,
            min_lr=Config.MIN_LR,
            verbose=1
        ),
        ModelCheckpoint(
            'best_model.keras',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            update_freq='epoch'
        )
    ]
    
    print("Training model...")
    # Calculate steps per epoch and validation steps
    steps_per_epoch = len(X_train) // Config.BATCH_SIZE
    validation_steps = len(X_val) // Config.BATCH_SIZE
    
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=Config.EPOCHS,
        callbacks=callbacks,
        verbose=1,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps
    )
    
    print("\nEvaluating model...")
    test_results = model.evaluate(val_dataset, verbose=1)
    print(f"\nTest Loss: {test_results[0]:.4f}")
    print(f"Test MSE: {test_results[1]:.4f}")
    print(f"Test RMSE: {test_results[2]:.4f}")
    
    # Plot training history
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(history.history['mse'], label='Training MSE')
    plt.plot(history.history['val_mse'], label='Validation MSE')
    plt.title('Model MSE')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(history.history['rmse'], label='Training RMSE')
    plt.plot(history.history['val_rmse'], label='Validation RMSE')
    plt.title('Model RMSE')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    print("\nSaving model and scalers...")
    model.save('6dof_ik_model.keras')
    import joblib
    joblib.dump(input_scaler, 'input_scaler.save')
    joblib.dump(output_scaler, 'output_scaler.save')
    
    # Analyze per-joint errors in batches
    print("\nCalculating per-joint errors...")
    joint_errors = []
    
    for x_batch, y_batch in val_dataset:
        y_pred_batch = model.predict(x_batch, verbose=0)
        y_pred_batch = output_scaler.inverse_transform(y_pred_batch)
        y_true_batch = output_scaler.inverse_transform(y_batch)
        joint_errors.append(np.abs(y_true_batch - y_pred_batch))
    
    joint_errors = np.concatenate(joint_errors, axis=0)
    
    plt.figure(figsize=(12, 6))
    plt.boxplot([joint_errors[:, i] for i in range(6)],
                labels=[f'Joint {i+1}' for i in range(6)])
    plt.title('Joint Prediction Error Distribution')
    plt.ylabel('Absolute Error (radians)')
    plt.grid(True)
    plt.show()
    
    print("\nPer-joint error statistics (radians):")
    for i in range(6):
        print(f"\nJoint {i+1}:")
        print(f"Mean error: {np.mean(joint_errors[:, i]):.6f}")
        print(f"Median error: {np.median(joint_errors[:, i]):.6f}")
        print(f"Max error: {np.max(joint_errors[:, i]):.6f}")

if __name__ == "__main__":
    main()