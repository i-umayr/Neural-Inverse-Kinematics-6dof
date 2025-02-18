import pyrealsense2 as rs
import numpy as np
import cv2
import mediapipe as mp
import time
from datetime import datetime
from scipy.spatial.transform import Rotation as R
from collections import deque
import tensorflow as tf
import joblib

# Robot dimensions (for reference/visualization)
ROBOT_ACTUAL_REACH = 1.3  # meters

# Dataset/Model workspace limits
WORKSPACE_LIMITS = {
    'x': (-1.0, 1.0),  # Normalized range matching dataset
    'y': (-1.0, 1.0),
    'z': (-1.0, 1.0)
}

def normalize_camera_coordinates(point_3d, camera_range=(-2, 2)):
    """Normalize camera coordinates to [-1, 1] range"""
    x, y, z = point_3d
    
    def normalize_coordinate(value, in_range, out_range=(-1, 1)):
        return np.interp(value, in_range, out_range)
    
    # Normalize each coordinate to [-1, 1]
    x_norm = normalize_coordinate(x, camera_range)
    y_norm = normalize_coordinate(y, camera_range)
    z_norm = normalize_coordinate(z, camera_range)
    
    return np.array([x_norm, y_norm, z_norm])

def calculate_hand_frame(landmarks):
    """Calculate hand orientation using anatomical coordinate system"""
    # Extract anatomical landmarks (already flipped for user perspective)
    wrist = np.array([
        -landmarks.landmark[0].x,
        landmarks.landmark[0].y,
        landmarks.landmark[0].z
    ])
    
    index_mcp = np.array([
        -landmarks.landmark[5].x,
        landmarks.landmark[5].y,
        landmarks.landmark[5].z
    ])
    
    middle_mcp = np.array([
        -landmarks.landmark[9].x,
        landmarks.landmark[9].y,
        landmarks.landmark[9].z
    ])
    
    pinky_mcp = np.array([
        -landmarks.landmark[17].x,
        landmarks.landmark[17].y,
        landmarks.landmark[17].z
    ])

    # Define axes
    y_axis = middle_mcp - wrist
    y_axis = y_axis / np.linalg.norm(y_axis)

    temp_x = index_mcp - pinky_mcp
    temp_x = temp_x / np.linalg.norm(temp_x)

    z_axis = np.cross(y_axis, temp_x)
    z_axis = z_axis / np.linalg.norm(z_axis)

    x_axis = np.cross(y_axis, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)

    # Create rotation matrix
    rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])

    # Convert to Euler angles in radians
    try:
        euler_angles = R.from_matrix(rotation_matrix).as_euler('xyz')
        return euler_angles
    except ValueError as e:
        print(f"Error computing Euler angles: {e}")
        return np.array([0.0, 0.0, 0.0])

def validate_normalized_pose(position, orientation):
    """Validate if normalized pose is within dataset limits"""
    x, y, z = position
    
    # Check position limits
    if not (WORKSPACE_LIMITS['x'][0] <= x <= WORKSPACE_LIMITS['x'][1] and
            WORKSPACE_LIMITS['y'][0] <= y <= WORKSPACE_LIMITS['y'][1] and
            WORKSPACE_LIMITS['z'][0] <= z <= WORKSPACE_LIMITS['z'][1]):
        return False
    
    # Check if point is within unit sphere (optional)
    distance_from_center = np.sqrt(x**2 + y**2 + z**2)
    if distance_from_center > 1.0:
        return False
        
    return True

def process_hand_pose(landmarks, depth_frame, intrinsics):
    """Process hand landmarks to get normalized pose"""
    # Get wrist position
    wrist = landmarks.landmark[0]
    x = int(wrist.x * depth_frame.width)
    y = int(wrist.y * depth_frame.height)
    depth = depth_frame.get_distance(x, y)
    
    # Convert to 3D point
    wrist_point_3d = rs.rs2_deproject_pixel_to_point(
        intrinsics, [x, y], depth
    )
    # Flip x-coordinate for user perspective
    wrist_point_3d = (-wrist_point_3d[0], wrist_point_3d[1], wrist_point_3d[2])
    
    # Normalize position to [-1, 1] range
    normalized_position = normalize_camera_coordinates(wrist_point_3d)
    
    # Calculate orientation in radians
    orientation = calculate_hand_frame(landmarks)
    
    return normalized_position, orientation

def convert_to_robot_coordinates(normalized_position):
    """Convert normalized position to robot coordinates if needed"""
    return normalized_position * ROBOT_ACTUAL_REACH

class HandTracker:
    def __init__(self, model_path='7dof_ik_model.keras', 
                 scaler_path='input_scaler.save'):
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8
        )
        
        # Initialize RealSense
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        
        # Load ML model and scaler
        try:
            self.model = tf.keras.models.load_model(model_path, compile=False)
            self.scaler = joblib.load(scaler_path)
            self.model_loaded = True
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model_loaded = False
        
        # Initialize buffers
        self.BUFFER_SIZE = 5
        self.position_buffer = deque(maxlen=self.BUFFER_SIZE)
        self.orientation_buffer = deque(maxlen=self.BUFFER_SIZE)
        
    def start(self):
        """Start the tracking pipeline"""
        self.pipeline.start(self.config)
        self.align = rs.align(rs.stream.color)
        
    def stop(self):
        """Stop the tracking pipeline"""
        self.pipeline.stop()
        self.hands.close()
        cv2.destroyAllWindows()
        
    def run(self):
        """Main tracking loop"""
        try:
            while True:
                # Get frames
                frames = self.pipeline.wait_for_frames()
                aligned_frames = self.align.process(frames)
                
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
                
                if not depth_frame or not color_frame:
                    continue
                
                # Get intrinsics for depth frame
                intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
                
                # Process image
                color_image = np.asanyarray(color_frame.get_data())
                results = self.hands.process(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
                
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Get normalized position and orientation
                        position, orientation = process_hand_pose(
                            hand_landmarks, depth_frame, intrinsics
                        )
                        
                        # Add to buffers
                        self.position_buffer.append(position)
                        self.orientation_buffer.append(orientation)
                        
                        if len(self.position_buffer) == self.BUFFER_SIZE:
                            # Calculate moving averages
                            avg_position = np.mean(self.position_buffer, axis=0)
                            avg_orientation = np.mean(self.orientation_buffer, axis=0)
                            
                            if validate_normalized_pose(avg_position, avg_orientation):
                                # Convert to robot coordinates for visualization
                                robot_position = convert_to_robot_coordinates(avg_position)
                                
                                # Prepare input for model
                                if self.model_loaded:
                                    # Here you would include current joint angles
                                    # For now, using zeros as placeholder
                                    current_joints = np.zeros(7)  
                                    model_input = np.concatenate([
                                        current_joints,
                                        avg_position,
                                        avg_orientation
                                    ])
                                    
                                    # Scale input
                                    model_input_scaled = self.scaler.transform(
                                        model_input.reshape(1, -1)
                                    )
                                    
                                    # Get prediction
                                    prediction = self.model.predict(
                                        model_input_scaled, verbose=0
                                    )
                                    
                                    # Display predictions
                                    cv2.putText(color_image, 
                                              f"Joint Predictions: {prediction[0][:3]}...", 
                                              (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 
                                              0.5, (0, 255, 0), 2)
                                
                                # Display positions
                                cv2.putText(color_image, 
                                          f"Norm Pos: {avg_position[0]:.2f}, {avg_position[1]:.2f}, {avg_position[2]:.2f}", 
                                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                                          1, (0, 255, 0), 2)
                                cv2.putText(color_image, 
                                          f"Robot Pos: {robot_position[0]:.2f}, {robot_position[1]:.2f}, {robot_position[2]:.2f}", 
                                          (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                                          1, (0, 255, 0), 2)
                            else:
                                cv2.putText(color_image, 
                                          "Hand position out of range!", 
                                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                                          1, (0, 0, 255), 2)
                
                # Display image
                cv2.imshow('Hand Tracking', color_image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        finally:
            self.stop()

def main():
    tracker = HandTracker()
    tracker.start()
    tracker.run()

if __name__ == "__main__":
    main()