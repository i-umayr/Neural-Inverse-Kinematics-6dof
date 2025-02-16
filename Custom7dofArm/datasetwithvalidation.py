import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from scipy.spatial.transform import Rotation

class EnhancedRobotVisualizer:
    def __init__(self):
        # DH parameters [a, alpha, d, theta_offset]
        self.dh_params = [
            [0, np.pi/2, 0.15, np.pi],        # Joint 1
            [0.435, -np.pi/2, 0, -np.pi/2],   # Joint 2
            [0, np.pi/2, 0.1025, 0],          # Joint 3
            [0.4193, -np.pi/2, 0, 0],         # Joint 4
            [0, np.pi/2, 0.1075, 0],          # Joint 5
            [0.091, -np.pi/2, 0, 0],          # Joint 6
            [0, 0, 0.0245, 0]                 # Joint 7
        ]
        
        self.joint_limits = {
            'q1': (0, 3.14),          # joint_1
            'q2': (-1.9199, 1.9199),  # Joint_2
            'q3': (0, 3.14),          # Joint_3
            'q4': (-0.2618, 3.4907),  # joint_4
            'q5': (0, 3.14),          # joint_5
            'q6': (-3.14, 0),         # joint_6
            'q7': (0, 3.14)           # Joint_7
        }

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
        """Calculate forward kinematics and return all link positions"""
        T = np.eye(4)
        link_positions = [(0, 0, 0)]  # Start with base position
        
        for i in range(7):
            a, alpha, d, theta_offset = self.dh_params[i]
            theta = joint_angles[i] + theta_offset
            
            T_i = self.dh_transform(a, alpha, d, theta)
            T = T @ T_i
            
            # Store each link's position
            position = T[:3, 3]
            link_positions.append(tuple(position))
        
        # Extract final orientation
        rotation_matrix = T[:3, :3]
        euler_angles = Rotation.from_matrix(rotation_matrix).as_euler('xyz')
        
        return link_positions, T[:3, 3], euler_angles

    def check_reachability(self, position, tolerance=0.01):
        """Check if a position is reachable by the robot"""
        # Calculate maximum reach
        max_reach = sum([np.sqrt(params[0]**2 + params[2]**2) for params in self.dh_params])
        
        # Check if position is within maximum reach
        distance = np.linalg.norm(position)
        if distance > max_reach:
            return False
        
        # Check minimum reach (assuming base offset)
        min_reach = self.dh_params[0][2]  # Base height
        if distance < min_reach:
            return False
            
        return True

    def visualize_robot_config(self, joint_angles, ax=None):
        """Visualize robot configuration with links"""
        if ax is None:
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')

        # Get link positions
        link_positions, end_effector, _ = self.forward_kinematics(joint_angles)
        link_positions = np.array(link_positions)
        
        # Plot links
        for i in range(len(link_positions)-1):
            x = [link_positions[i][0], link_positions[i+1][0]]
            y = [link_positions[i][1], link_positions[i+1][1]]
            z = [link_positions[i][2], link_positions[i+1][2]]
            ax.plot(x, y, z, 'b-', linewidth=2)
        
        # Plot joints
        ax.scatter(link_positions[:-1, 0], link_positions[:-1, 1], 
                  link_positions[:-1, 2], c='r', marker='o')
        
        # Plot end-effector
        ax.scatter(link_positions[-1, 0], link_positions[-1, 1], 
                  link_positions[-1, 2], c='g', marker='s')
        
        # Set equal aspect ratio
        ax.set_box_aspect([1,1,1])
        return ax

    def visualize_workspace_with_robot(self, dataset_path, num_configs=5):
        """Visualize workspace points and sample robot configurations"""
        data = pd.read_csv(dataset_path)
        
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot workspace points
        scatter = ax.scatter(data['x_desired'], data['y_desired'], data['z_desired'],
                           c=data['q1_desired'], cmap='viridis', alpha=0.1, s=1)
        
        # Plot sample configurations
        for i in range(num_configs):
            joint_angles = [
                data[f'q{j}_desired'].iloc[i] for j in range(1, 8)
            ]
            self.visualize_robot_config(joint_angles, ax)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        plt.colorbar(scatter, label='q1 angle (rad)')
        plt.title('Robot Workspace with Sample Configurations')
        plt.show()

    def validate_dataset(self, dataset_path):
        """Validate reachability of positions in dataset"""
        data = pd.read_csv(dataset_path)
        total_points = len(data)
        reachable_points = 0
        
        print("Validating dataset reachability...")
        
        for i in range(total_points):
            position = np.array([
                data['x_desired'].iloc[i],
                data['y_desired'].iloc[i],
                data['z_desired'].iloc[i]
            ])
            
            if self.check_reachability(position):
                reachable_points += 1
                
        print(f"\nReachability Analysis:")
        print(f"Total points: {total_points}")
        print(f"Reachable points: {reachable_points}")
        print(f"Percentage reachable: {(reachable_points/total_points)*100:.2f}%")
        
        return reachable_points/total_points

def main():
    visualizer = EnhancedRobotVisualizer()
    
    # Validate and visualize dataset
    dataset_path = 'robot_dataset.csv'
    reachability_ratio = visualizer.validate_dataset(dataset_path)
    
    # Visualize workspace with robot configurations
    visualizer.visualize_workspace_with_robot(dataset_path)
    
if __name__ == "__main__":
    main()