import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

class DatasetVisualizer:
    def __init__(self, csv_path):
        """Initialize visualizer with dataset path"""
        self.data = pd.read_csv(csv_path)
        self.joint_names = [f'q{i}' for i in range(1, 7)]  # Changed to 6 joints
        
    def plot_workspace(self):
        """Plot 3D workspace of end-effector positions"""
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(
            self.data['x_desired'],
            self.data['y_desired'],
            self.data['z_desired'],
            c=self.data['q1_desired'],
            cmap='viridis',
            alpha=0.6
        )
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('Robot Workspace - End Effector Positions')
        
        plt.colorbar(scatter, label='q1 angle (rad)')
        plt.tight_layout()
        plt.show()
        
    def plot_joint_distributions(self):
        """Plot distributions of joint angles"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))  # Changed to 2x3 grid
        axes = axes.flatten()
        
        for i, joint in enumerate(self.joint_names):
            current_col = f'{joint}_current'
            desired_col = f'{joint}_desired'
            
            sns.kdeplot(data=self.data, x=current_col, ax=axes[i], label='Current')
            sns.kdeplot(data=self.data, x=desired_col, ax=axes[i], label='Desired')
            
            axes[i].set_title(f'{joint} Distribution')
            axes[i].set_xlabel('Angle (rad)')
            axes[i].legend()
        
        plt.tight_layout()
        plt.show()
        
    def plot_joint_correlations(self):
        """Plot correlation matrix between joint angles"""
        desired_joints = [f'{joint}_desired' for joint in self.joint_names]
        correlation_matrix = self.data[desired_joints].corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            correlation_matrix,
            annot=True,
            cmap='coolwarm',
            center=0,
            fmt='.2f',
            square=True
        )
        plt.title('Joint Angle Correlations')
        plt.tight_layout()
        plt.show()
        
    def plot_position_vs_angles(self):
        """Plot relationship between end-effector position and joint angles"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))  # Changed to 2x3 grid
        axes = axes.flatten()
        
        # Plot each position component against each joint angle
        positions = ['x_desired', 'y_desired', 'z_desired']
        colors = ['r', 'g', 'b']
        
        for i, pos in enumerate(positions):
            ax1 = axes[i]
            ax2 = axes[i+3]
            
            # Plot against q1 and q2
            ax1.scatter(self.data['q1_desired'], self.data[pos], 
                       alpha=0.1, c=colors[i], label='q1')
            ax1.scatter(self.data['q2_desired'], self.data[pos], 
                       alpha=0.1, c='gray', label='q2')
            ax1.set_xlabel('Joint Angle (rad)')
            ax1.set_ylabel(f'{pos} (m)')
            ax1.legend()
            
            # Plot against q3 and q4
            ax2.scatter(self.data['q3_desired'], self.data[pos], 
                       alpha=0.1, c=colors[i], label='q3')
            ax2.scatter(self.data['q4_desired'], self.data[pos], 
                       alpha=0.1, c='gray', label='q4')
            ax2.set_xlabel('Joint Angle (rad)')
            ax2.set_ylabel(f'{pos} (m)')
            ax2.legend()
        
        plt.tight_layout()
        plt.show()
        
    def plot_orientation_distribution(self):
        """Plot distribution of end-effector orientations"""
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(
            self.data['roll'],
            self.data['pitch'],
            self.data['yaw'],
            c=self.data['z_desired'],
            cmap='viridis',
            alpha=0.6
        )
        
        ax.set_xlabel('Roll (rad)')
        ax.set_ylabel('Pitch (rad)')
        ax.set_zlabel('Yaw (rad)')
        ax.set_title('End Effector Orientation Distribution')
        
        plt.colorbar(scatter, label='Z position (m)')
        plt.tight_layout()
        plt.show()
        
    def print_statistics(self):
        """Print basic statistics about the dataset"""
        print("\nDataset Statistics:")
        print("-" * 50)
        
        # Joint angle ranges
        print("\nJoint Angle Ranges (radians):")
        for joint in self.joint_names:
            desired_col = f'{joint}_desired'
            min_val = self.data[desired_col].min()
            max_val = self.data[desired_col].max()
            print(f"{joint}: [{min_val:.2f}, {max_val:.2f}]")
            
        # Workspace dimensions
        print("\nWorkspace Dimensions (meters):")
        for pos in ['x_desired', 'y_desired', 'z_desired']:
            min_val = self.data[pos].min()
            max_val = self.data[pos].max()
            print(f"{pos}: [{min_val:.2f}, {max_val:.2f}]")
            
        # Orientation ranges
        print("\nOrientation Ranges (radians):")
        for orient in ['roll', 'pitch', 'yaw']:
            min_val = self.data[orient].min()
            max_val = self.data[orient].max()
            print(f"{orient}: [{min_val:.2f}, {max_val:.2f}]")

def main():
    # Create visualizer instance
    visualizer = DatasetVisualizer('robot_dataset.csv')
    
    # Print basic statistics
    visualizer.print_statistics()
    
    # Generate all plots
    print("\nGenerating visualization plots...")
    visualizer.plot_workspace()
    visualizer.plot_joint_distributions()
    visualizer.plot_joint_correlations()
    visualizer.plot_position_vs_angles()
    visualizer.plot_orientation_distribution()

if __name__ == "__main__":
    main()