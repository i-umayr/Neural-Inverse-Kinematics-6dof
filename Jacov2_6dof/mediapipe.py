import pyrealsense2 as rs
import numpy as np
import cv2
import mediapipe as mp
import csv
import time
from datetime import datetime
from scipy.spatial.transform import Rotation as R
from collections import deque

# Workspace limits matching the dataset
WORKSPACE_LIMITS = {
    'x': (-1.0, 1.0),
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
    # Extract anatomical landmarks with x-coordinate flipped for user perspective
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

def validate_pose(position):
    """Validate if normalized pose is within workspace limits"""
    x, y, z = position
    return (WORKSPACE_LIMITS['x'][0] <= x <= WORKSPACE_LIMITS['x'][1] and
            WORKSPACE_LIMITS['y'][0] <= y <= WORKSPACE_LIMITS['y'][1] and
            WORKSPACE_LIMITS['z'][0] <= z <= WORKSPACE_LIMITS['z'][1])

def main():
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.8,
        min_tracking_confidence=0.8
    )

    # Initialize RealSense Pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    # Start streaming
    pipeline.start(config)
    align = rs.align(rs.stream.color)

    # Initialize buffers for moving average
    BUFFER_SIZE = 5
    position_buffer = deque(maxlen=BUFFER_SIZE)
    orientation_buffer = deque(maxlen=BUFFER_SIZE)

    # Create CSV file for logging
    csv_filename = f'hand_poses_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    with open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['timestamp', 'x', 'y', 'z', 'roll', 'pitch', 'yaw'])

        try:
            while True:
                # Get frames
                frames = pipeline.wait_for_frames()
                aligned_frames = align.process(frames)
                
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
                
                if not depth_frame or not color_frame:
                    continue

                # Process color image
                color_image = np.asanyarray(color_frame.get_data())
                results = hands.process(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Get wrist position
                        wrist = hand_landmarks.landmark[0]
                        x = int(wrist.x * color_image.shape[1])
                        y = int(wrist.y * color_image.shape[0])
                        depth = depth_frame.get_distance(x, y)
                        
                        # Convert to 3D point
                        wrist_point_3d = rs.rs2_deproject_pixel_to_point(
                            depth_frame.profile.as_video_stream_profile().intrinsics,
                            [x, y], depth
                        )
                        # Flip x-coordinate for user perspective
                        wrist_point_3d = (-wrist_point_3d[0], wrist_point_3d[1], wrist_point_3d[2])
                        
                        # Normalize position
                        normalized_position = normalize_camera_coordinates(wrist_point_3d)
                        
                        # Calculate orientation
                        orientation = calculate_hand_frame(hand_landmarks)
                        
                        # Add to buffers
                        position_buffer.append(normalized_position)
                        orientation_buffer.append(orientation)
                        
                        # Calculate moving averages when buffer is full
                        if len(position_buffer) == BUFFER_SIZE:
                            avg_position = np.mean(position_buffer, axis=0)
                            avg_orientation = np.mean(orientation_buffer, axis=0)
                            
                            if validate_pose(avg_position):
                                # Log to CSV
                                csv_writer.writerow([
                                    time.time(),
                                    avg_position[0],
                                    avg_position[1],
                                    avg_position[2],
                                    avg_orientation[0],
                                    avg_orientation[1],
                                    avg_orientation[2]
                                ])
                                csvfile.flush()
                                
                                # Display values
                                cv2.putText(color_image, 
                                          f"Position: {avg_position[0]:.2f}, {avg_position[1]:.2f}, {avg_position[2]:.2f}", 
                                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                                cv2.putText(color_image, 
                                          f"Rotation: {np.rad2deg(avg_orientation[0]):.1f}, {np.rad2deg(avg_orientation[1]):.1f}, {np.rad2deg(avg_orientation[2]):.1f}", 
                                          (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            else:
                                cv2.putText(color_image, 
                                          "Hand out of workspace!", 
                                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Display the image
                cv2.imshow('Hand Tracking', color_image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            pipeline.stop()
            cv2.destroyAllWindows()
            hands.close()

if __name__ == "__main__":
    main()