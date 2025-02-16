import pybullet as p

# Connect to PyBullet
physicsClient = p.connect(p.DIRECT)  # Use p.DIRECT for non-graphical version

# Load your robot
robot_id = p.loadURDF('final_7dof_urdf_9oct2024/urdf/final_7dof_urdf_9oct2024.urdf', useFixedBase=True)

# Get the number of joints
num_joints = p.getNumJoints(robot_id)

# Extract joint limits
joint_limits = []
for i in range(num_joints):
    joint_info = p.getJointInfo(robot_id, i)
    joint_name = joint_info[1].decode('utf-8')  # Joint name
    joint_type = joint_info[2]  # Joint type (0 = revolute, 1 = prismatic, etc.)
    lower_limit = joint_info[8]  # Lower limit
    upper_limit = joint_info[9]  # Upper limit
    
    if joint_type == p.JOINT_REVOLUTE or joint_type == p.JOINT_PRISMATIC:
        joint_limits.append((joint_name, lower_limit, upper_limit))

# Print joint limits
for name, lower, upper in joint_limits:
    print(f"Joint: {name}, Lower Limit: {lower}, Upper Limit: {upper}")

# Disconnect from PyBullet
p.disconnect()