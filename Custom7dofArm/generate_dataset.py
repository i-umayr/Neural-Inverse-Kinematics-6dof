import pybullet as p
import numpy as np
import pandas as pd

# Connect to PyBullet
physicsClient = p.connect(p.DIRECT)  # Use p.DIRECT for non-graphical version

# Load your robot
robot_id = p.loadURDF('final_7dof_urdf_9oct2024/urdf/final_7dof_urdf_9oct2024.urdf', useFixedBase=True)

# Define joint limits (from your output)
joint_limits = [
    (0.0, 3.14),        # Joint 1
    (-1.9199, 1.9199),  # Joint 2
    (0.0, 3.14),        # Joint 3
    (-0.2618, 3.4907),  # Joint 4
    (0.0, 3.14),        # Joint 5
    (-3.14, 0.0),       # Joint 6
    (0.0, 3.14)         # Joint 7
]

# Number of samples in the dataset
num_samples = 10000

# Generate random joint angles
joint_angles = np.random.uniform(
    low=[limit[0] for limit in joint_limits],
    high=[limit[1] for limit in joint_limits],
    size=(num_samples, len(joint_limits))
)

# Function to compute forward kinematics
def compute_forward_kinematics(joint_angles):
    # Reset the robot to the given joint angles
    for i, angle in enumerate(joint_angles):
        p.resetJointState(robot_id, i, angle)
    
    # Get the end-effector state
    end_effector_index = 7  # Replace with the index of your end-effector joint
    state = p.getLinkState(robot_id, end_effector_index)
    position = state[0]  # End-effector position (x, y, z)
    orientation = p.getEulerFromQuaternion(state[1])  # End-effector orientation (yaw, pitch, roll)
    
    return position, orientation

# Compute end-effector states for all joint angle configurations
end_effector_states = np.array([compute_forward_kinematics(angles) for angles in joint_angles])

# Disconnect from PyBullet
p.disconnect()

# Extract positions and orientations
positions = end_effector_states[:, 0]
orientations = end_effector_states[:, 1]

# Create a DataFrame for the dataset
data = pd.DataFrame({
    'x': [pos[0] for pos in positions],
    'y': [pos[1] for pos in positions],
    'z': [pos[2] for pos in positions],
    'yaw': [orient[0] for orient in orientations],
    'pitch': [orient[1] for orient in orientations],
    'roll': [orient[2] for orient in orientations],
    'q1': joint_angles[:, 0],
    'q2': joint_angles[:, 1],
    'q3': joint_angles[:, 2],
    'q4': joint_angles[:, 3],
    'q5': joint_angles[:, 4],
    'q6': joint_angles[:, 5],
    'q7': joint_angles[:, 6]
})

# Save the dataset to a CSV file
data.to_csv('robot_dataset.csv', index=False)

print("Dataset generated and saved to 'robot_dataset.csv'.")