import pybullet as p

# Connect to PyBullet
physicsClient = p.connect(p.DIRECT)

# Load the URDF file
robot_id = p.loadURDF('final_7dof_urdf_9oct2024/urdf/final_7dof_urdf_9oct2024.urdf', useFixedBase=True)

# Get joint and link information
num_joints = p.getNumJoints(robot_id)
for i in range(num_joints):
    joint_info = p.getJointInfo(robot_id, i)
    print(f"Joint Index: {i}, Name: {joint_info[1]}, Type: {joint_info[2]}")