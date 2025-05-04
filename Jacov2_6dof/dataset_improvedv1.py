import numpy as np
from scipy.spatial.transform import Rotation
import pandas as pd
import random

class RobotDatasetGenerator:
    def __init__(self):
        # Joint limits from URDF
        self.joint_limits = {
            'q1': (-2*np.pi, 2*np.pi),          # joint_1 (continuous)
            'q2': (47/180*np.pi, 313/180*np.pi), # joint_2
            'q3': (19/180*np.pi, 341/180*np.pi), # joint_3
            'q4': (-2*np.pi, 2*np.pi),          # joint_4 (continuous)
            'q5': (-2*np.pi, 2*np.pi),          # joint_5 (continuous)
            'q6': (-2*np.pi, 2*np.pi)           # joint_6 (continuous)
        }
        
        # DH Parameters [a, alpha, d, theta]
        self.dh_params = [
            [0, np.pi/2, 0.15675, np.pi],       # Joint 1
            [0, -np.pi/2, -0.11875, -np.pi/2],  # Joint 2
            [0.410, np.pi, 0, np.pi],           # Joint 3
            [0, -np.pi/2, -0.21033, np.pi],     # Joint 4
            [0, np.pi/3, -0.07414, np.pi],      # Joint 5
            [0, 0, -0.16, np.pi/2]              # Joint 6 (including end effector)
        ]
        
        # Number of joints
        self.num_joints = len(self.dh_params)

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
        transforms = []  # Store all transformations
        
        for i in range(self.num_joints):
            # Update theta in DH parameters with joint angle
            a, alpha, d, theta_offset = self.dh_params[i]
            theta = joint_angles[i] + theta_offset
            
            # Calculate transformation matrix for this joint
            T_i = self.dh_transform(a, alpha, d, theta)
            T = T @ T_i
            transforms.append(T.copy())  # Save current transform
        
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

    def generate_nearby_config(self, base_config, max_delta=0.5):
        """Generate a configuration near the base configuration with larger differences
        
        Args:
            base_config: Base joint configuration
            max_delta: Maximum change in joint angle (increased from 0.1 to 0.5)
        """
        new_config = []
        for i, angle in enumerate(base_config):
            min_val, max_val = self.joint_limits[f'q{i+1}']
            delta = random.uniform(-max_delta, max_delta)
            new_angle = np.clip(angle + delta, min_val, max_val)
            new_config.append(new_angle)
        return np.array(new_config)
    
    def generate_varying_config(self, base_config, delta_range=(0.1, 1.0)):
        """Generate a configuration with varying differences from the base
        
        Args:
            base_config: Base joint configuration
            delta_range: Tuple of (min_delta, max_delta) for joint angle changes
        """
        min_delta, max_delta = delta_range
        new_config = []
        for i, angle in enumerate(base_config):
            min_val, max_val = self.joint_limits[f'q{i+1}']
            
            # Generate a random delta within the specified range
            # Use a distribution that favors smaller deltas but allows larger ones
            delta_magnitude = random.uniform(min_delta, max_delta)
            delta = delta_magnitude * random.choice([-1, 1])
            
            new_angle = np.clip(angle + delta, min_val, max_val)
            new_config.append(new_angle)
        return np.array(new_config)

    def analyze_workspace(self, num_samples=10000):
        """Analyze the workspace by sampling random configurations"""
        x_min, x_max = float('inf'), float('-inf')
        y_min, y_max = float('inf'), float('-inf')
        z_min, z_max = float('inf'), float('-inf')
        
        for _ in range(num_samples):
            config = self.generate_random_config()
            position, _ = self.forward_kinematics(config)
            
            x_min = min(x_min, position[0])
            x_max = max(x_max, position[0])
            y_min = min(y_min, position[1])
            y_max = max(y_max, position[1])
            z_min = min(z_min, position[2])
            z_max = max(z_max, position[2])
        
        return {
            'x_range': (x_min, x_max),
            'y_range': (y_min, y_max),
            'z_range': (z_min, z_max)
        }

    def generate_dataset(self, num_samples=100000):
        """Generate dataset with current and desired joint angles with variable differences
        
        This updated method creates a more diverse dataset with varying degrees of difference
        between current and desired configurations, better representing real-world conditions
        where pipeline latency causes larger differences.
        """
        data = []
        
        # Calculate maximum reach
        max_reach = sum([abs(params[0]) + abs(params[2]) for params in self.dh_params])
        print(f"Maximum theoretical reach: {max_reach:.3f} meters")
        
        # Generate distribution of samples with different magnitudes of change
        small_diff_samples = int(num_samples * 0.3)  # 30% small differences (0.1-0.3)
        medium_diff_samples = int(num_samples * 0.4)  # 40% medium differences (0.3-0.6)
        large_diff_samples = num_samples - small_diff_samples - medium_diff_samples  # 30% large differences (0.6-1.0)
        
        print(f"Generating {small_diff_samples} samples with small differences (0.1-0.3 radians)")
        for _ in range(small_diff_samples):
            # Generate desired joint configuration
            desired_config = self.generate_random_config()
            
            # Generate current configuration with small difference from desired
            current_config = self.generate_varying_config(desired_config, delta_range=(0.1, 0.3))
            
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
                'q6_desired': desired_config[5]
            }
            
            data.append(sample)
        
        print(f"Generating {medium_diff_samples} samples with medium differences (0.3-0.6 radians)")
        for _ in range(medium_diff_samples):
            desired_config = self.generate_random_config()
            current_config = self.generate_varying_config(desired_config, delta_range=(0.3, 0.6))
            position, euler_angles = self.forward_kinematics(desired_config)
            
            sample = {
                'q1_current': current_config[0],
                'q2_current': current_config[1],
                'q3_current': current_config[2],
                'q4_current': current_config[3],
                'q5_current': current_config[4],
                'q6_current': current_config[5],
                'x_desired': position[0],
                'y_desired': position[1],
                'z_desired': position[2],
                'roll': euler_angles[0],
                'pitch': euler_angles[1],
                'yaw': euler_angles[2],
                'q1_desired': desired_config[0],
                'q2_desired': desired_config[1],
                'q3_desired': desired_config[2],
                'q4_desired': desired_config[3],
                'q5_desired': desired_config[4],
                'q6_desired': desired_config[5]
            }
            
            data.append(sample)
            
        print(f"Generating {large_diff_samples} samples with large differences (0.6-1.0 radians)")
        for _ in range(large_diff_samples):
            desired_config = self.generate_random_config()
            current_config = self.generate_varying_config(desired_config, delta_range=(0.6, 1.0))
            position, euler_angles = self.forward_kinematics(desired_config)
            
            sample = {
                'q1_current': current_config[0],
                'q2_current': current_config[1],
                'q3_current': current_config[2],
                'q4_current': current_config[3],
                'q5_current': current_config[4],
                'q6_current': current_config[5],
                'x_desired': position[0],
                'y_desired': position[1],
                'z_desired': position[2],
                'roll': euler_angles[0],
                'pitch': euler_angles[1],
                'yaw': euler_angles[2],
                'q1_desired': desired_config[0],
                'q2_desired': desired_config[1],
                'q3_desired': desired_config[2],
                'q4_desired': desired_config[3],
                'q5_desired': desired_config[4],
                'q6_desired': desired_config[5]
            }
            
            data.append(sample)
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Add metadata column to indicate difference magnitude
        differences = []
        for i in range(len(df)):
            current = np.array([df.iloc[i][f'q{j}_current'] for j in range(1, 7)])
            desired = np.array([df.iloc[i][f'q{j}_desired'] for j in range(1, 7)])
            max_diff = np.max(np.abs(current - desired))
            differences.append(max_diff)
        
        df['max_joint_diff'] = differences
        
        # Print statistics on the generated dataset
        print("\nDataset statistics:")
        print(f"Average max joint difference: {np.mean(differences):.4f} radians")
        print(f"Median max joint difference: {np.median(differences):.4f} radians")
        print(f"Min max joint difference: {np.min(differences):.4f} radians")
        print(f"Max max joint difference: {np.max(differences):.4f} radians")
        
        return df

def main():
    # Create generator and generate dataset
    generator = RobotDatasetGenerator()
    
    # Analyze workspace first (optional)
    workspace = generator.analyze_workspace(num_samples=5000)
    print("Workspace analysis:")
    print(f"X range: {workspace['x_range'][0]:.3f} to {workspace['x_range'][1]:.3f} meters")
    print(f"Y range: {workspace['y_range'][0]:.3f} to {workspace['y_range'][1]:.3f} meters")
    print(f"Z range: {workspace['z_range'][0]:.3f} to {workspace['z_range'][1]:.3f} meters")
    
    # Generate more diverse dataset
    dataset = generator.generate_dataset(num_samples=100000)
    
    # Save dataset
    dataset.to_csv('robot_dataset_diverse.csv', index=False)
    print("\nDataset generated and saved to 'robot_dataset_diverse.csv'")
    print("\nDataset shape:", dataset.shape)
    print("\nSample of the dataset:")
    print(dataset.head())
    
    # Optional: Generate a histogram of joint differences
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.hist(dataset['max_joint_diff'], bins=30)
        plt.title('Distribution of Maximum Joint Differences')
        plt.xlabel('Maximum Joint Difference (radians)')
        plt.ylabel('Count')
        plt.savefig('joint_differences_histogram.png')
        print("\nHistogram of joint differences saved to 'joint_differences_histogram.png'")
    except ImportError:
        print("Matplotlib not available - skipping histogram generation")

if __name__ == "__main__":
    main()