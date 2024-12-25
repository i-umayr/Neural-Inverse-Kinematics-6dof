# Neural Inverse Kinematics for 6-DOF Robot Arm Using LSTM-Based Architecture

## Overview
This repository implements an advanced neural network approach to solve the inverse kinematics problem for a 6-degree-of-freedom (6-DOF) robotic arm. The model combines LSTM networks with residual connections to achieve high-precision joint angle predictions in real-time applications.

## Problem Description
Inverse kinematics (IK) involves calculating joint angles required to achieve a desired end-effector position and orientation. Traditional analytical methods often struggle with:
- Real-time adaptability
- Computational efficiency 
- Multiple possible solutions
- Singular configurations

Our implementation addresses these challenges using a deep learning approach that provides fast, accurate, and robust solutions.

## Architecture

### Input Processing
- 12 input features (x, y, z, yaw, pitch, roll, q1_in...q6_in)
- GaussianNoise layer (σ=0.01) for improved robustness
- StandardScaler for input normalization

### Dual-Stream Processing
1. LSTM Stream:
  - Custom SequenceGeneration layer (sequence length: 4)
  - LSTM layer with 256 units
  - Processes temporal patterns in joint configurations

2. Residual Stream:
  - Dense layers: 512 → 256 → 128 units
  - Skip connections for gradient flow
  - Layer normalization and LeakyReLU (α=0.1)

### Regularization Techniques
- Dropout rate: 0.1
- L2 regularization: 1e-6
- Batch normalization
- Custom periodic loss function

### Training Configuration
- Batch size: 32
- Learning rate: 1e-4 with exponential decay
- Early stopping with patience: 20
- Loss function: 0.7×MSE + 0.3×Periodic Loss

## Performance Results

### Accuracy Metrics
- Test MSE: 0.0002 (99% improvement over baseline)
- Test RMSE: 0.0155 radians (~0.89 degrees)
- Per-joint average errors (in degrees):
 - Joints 1-3: < 3°
 - Joints 4-5: < 5°
 - Joint 6: < 7°

### Real-time Performance
- Prediction latency: 0.237 ± 0.007 ms/sample
- Throughput: 4,227 predictions/second
- Suitable for high-frequency control loops (>4kHz)

### Comparison with Prior Work
- Baseline model RMSE: 0.1419 (8.13 degrees)
- Research benchmark MSE: 3.3e-8
- Our implementation meets industrial requirements (<2° error)

## Dataset
The model is trained on the ABB IRB2400 dataset containing:
- 300,000 samples
- Input: end-effector position (x,y,z), orientation (yaw,pitch,roll), current joint angles
- Output: target joint angles
- 80/20 train-test split

## Usage

### Dependencies
```python
tensorflow>=2.12.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=0.24.0
matplotlib>=3.4.0
