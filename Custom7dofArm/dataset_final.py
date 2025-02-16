import numpy as np
from scipy.spatial.transform import Rotation
import pandas as pd
import random

class RobotDatasetGenerator:
    def __init__(self):
        # Joint limits from URDF
        self.joint_limits = {
            'q1': (0, 3.14),          # joint_1 (continuous)
            'q2': (-1.9199, 1.9199),  # Joint_2
            'q3': (0, 3.14),          # Joint_3 (continuous)
            'q4': (-0.2618, 3.4907),  # joint_4
            'q5': (0, 3.14),          # joint_5 (continuous)
            'q6': (-3.14, 0),         # joint_6
            'q7': (0, 3.14)           # Joint_7 (continuous)
        }
        
        # DH Parameters (to be filled based on your robot)
        # Format: [a, alpha, d, theta]
        self.dh_params = [
            [0, np.pi/2, 0.05, np.pi],    # Joint 1
            [0.03, np.pi/2, 0.1075, 0],   # Joint 2
            [0.435, 0, 0.0294, np.pi],    # Joint 3
            [0.03, np.pi/2, 0.1025, 0],   # Joint 4
            [0.4193, np.pi/2, -0.0295, 0], # Joint 5
            [0.024, np.pi/2, 0.0675, 0.0507], # Joint 6
            [0.091, -np.pi/2, -0.0245, -np.pi/2] # Joint 7
        ]

    def dh_transform(self, a, alpha, d, theta):
        """Calculate transformation matrix using DH parameters"""
        ct = np.cos(theta)
        st = np.sin(theta)
        ca = np.cos(alpha)
        sa = np.sin(alpha)
        
        return np.array([
            [ct, -st*ca, st*sa, a*ct],
            [st, ct*ca, -ct*sa, a*st],
            [0, sa, ca, d],
            [0, 0, 0, 1]
        ])

    def forward_kinematics(self, joint_angles):
        """Calculate forward kinematics using DH parameters"""
        T = np.eye(4)
        
        for i in range(7):
            # Update theta in DH parameters with joint angle
            a, alpha, d, _ = self.dh_params[i]
            theta = joint_angles[i]
            
            # Calculate transformation matrix for this joint
            T_i = self.dh_transform(a, alpha, d, theta)
            T = T @ T_i
        
        # Extract position and orientation
        position = T[:3, 3]
        rotation_matrix = T[:3, :3]
        euler_angles = Rotation.from_matrix(rotation_matrix).as_euler('xyz')
        
        return position, euler_angles

    def generate_random_config(self):
        """Generate random joint configuration within limits"""
        config = []
        for joint, (min_val, max_val) in self.joint_limits.items():
            angle = random.uniform(min_val, max_val)
            config.append(angle)
        return np.array(config)

    def generate_nearby_config(self, base_config, max_delta=0.1):
        """Generate a configuration near the base configuration"""
        new_config = []
        for i, angle in enumerate(base_config):
            min_val, max_val = self.joint_limits[f'q{i+1}']
            delta = random.uniform(-max_delta, max_delta)
            new_angle = np.clip(angle + delta, min_val, max_val)
            new_config.append(new_angle)
        return np.array(new_config)

    def generate_dataset(self, num_samples=10000):
        """Generate dataset with current and desired joint angles"""
        data = []
        
        for _ in range(num_samples):
            # Generate desired joint configuration
            desired_config = self.generate_random_config()
            
            # Generate current configuration (slightly different from desired)
            current_config = self.generate_nearby_config(desired_config)
            
            # Calculate end-effector position and orientation for desired config
            position, euler_angles = self.forward_kinematics(desired_config)
            
            # Create data sample
            sample = {
                # Current joint angles (t-1)
                'q1_current': current_config[0],
                'q2_current': current_config[1],
                'q3_current': current_config[2],
                'q4_current': current_config[3],
                'q5_current': current_config[4],
                'q6_current': current_config[5],
                'q7_current': current_config[6],
                
                # Desired end-effector position
                'x_desired': position[0],
                'y_desired': position[1],
                'z_desired': position[2],
                
                # Desired end-effector orientation
                'roll': euler_angles[0],
                'pitch': euler_angles[1],
                'yaw': euler_angles[2],
                
                # Desired joint angles (output)
                'q1_desired': desired_config[0],
                'q2_desired': desired_config[1],
                'q3_desired': desired_config[2],
                'q4_desired': desired_config[3],
                'q5_desired': desired_config[4],
                'q6_desired': desired_config[5],
                'q7_desired': desired_config[6]
            }
            
            data.append(sample)
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        return df

def main():
    # Create generator and generate dataset
    generator = RobotDatasetGenerator()
    dataset = generator.generate_dataset(num_samples=10000)
    
    # Save dataset
    dataset.to_csv('robot_dataset.csv', index=False)
    print("Dataset generated and saved to 'robot_dataset.csv'")
    print("\nDataset shape:", dataset.shape)
    print("\nSample of the dataset:")
    print(dataset.head())

if __name__ == "__main__":
    main()