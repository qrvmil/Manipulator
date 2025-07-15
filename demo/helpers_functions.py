import numpy as np
import mujoco
from collections import deque


def print_robot_state(data, step=None):
    if step is not None:
        print(f"\n--- step {step} ---")
    
    print("data.ctrl")
    ctrl_names = ['waist', 'shoulder', 'elbow', 'forearm_roll', 'wrist_angle', 'wrist_rotate', 'gripper']
    for i, name in enumerate(ctrl_names):
        if i < 6:  
            print(f"  ctrl[{i}] {name:12}: {np.degrees(data.ctrl[i]):7.1f}° (target)")
        else:  
            print(f"  ctrl[{i}] {name:12}: {data.ctrl[i]:7.4f} m (target)")
    
    print("data.qpos")
    for i, name in enumerate(ctrl_names):
        if i < 6:  
            current = np.degrees(data.qpos[i])
            target = np.degrees(data.ctrl[i])
            error = target - current
            print(f"  qpos[{i}] {name:12}: {current:7.1f}° (error: {error:+6.1f}°)")
        else:  
            current = data.qpos[6]  
            target = data.ctrl[i]
            error = target - current
            print(f"  qpos[{i}] {name:12}: {current:7.4f} m (error: {error:+7.4f} m)")


def print_robot_info(model, data):
    """Выводит информацию о роботе"""
    print(f"Количество суставов: {model.nq}")
    print(f"Количество приводов: {model.nu}")
    
    print("\nДоступные сайты:")
    for i in range(model.nsite):
        site_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i)
        if site_name:
            print(f"  {i}: {site_name}")

def get_current_pose(model, data, site_name):
    """Получает текущую позицию и ориентацию сайта"""
    mujoco.mj_forward(model, data)
    
    
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)

    pos = data.site_xpos[site_id].copy()
    mat = data.site_xmat[site_id].copy().reshape(3, 3)
    
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, mat.flatten())
    
    return pos, quat


def get_robot_geom_ids(model, root_body_name):
    """
    Returns a list of geometry IDs that lie in the body subtree,
    the root of which is a body named root_body_name.
    Args:
        model: MjModel.
        root_body_name: name of the root body of the robot in the model (str).
    Returns:
        List[int]: ID of all geometries attached to the robot bodies.
    """
    root_id = model.body(root_body_name).id
    
    nbody = model.nbody
    parent = model.body_parentid
    children = [[] for _ in range(nbody)]
    for b in range(nbody):
        p = int(parent[b])
        if 0 <= p < nbody:
            children[p].append(b)

    # BFS from root_id: collect all robot bodies
    visited = set([root_id])
    queue = deque([root_id])
    robot_bodies = []

    while queue:
        b = queue.popleft()
        robot_bodies.append(b)
        for c in children[b]:
            if c not in visited:
                visited.add(c)
                queue.append(c)

    # For each body, get the geometry
    geom_ids = []
    for b in robot_bodies:
        adr   = model.body_geomadr[b]   # first geom-index
        cnt   = model.body_geomnum[b]   # how many geom's there
        geom_ids.extend(range(adr, adr + cnt))

    return geom_ids


def forward_kinematics(model, joint_angles, site_name='attachment_site'):
    """
    Calculates forward kinematics: by joint angles determines the position of the end-effector
    
    Args:
        model: MuJoCo model
        joint_angles: joint angles (list or numpy array of 7 elements for KUKA iiwa14)
        site_name: name of the end-effector site (default: 'attachment_site')
    
    Returns:
        tuple: (position, quaternion) 
            - position: numpy array [x, y, z] in meters
            - quaternion: numpy array [w, x, y, z] (quaternion orientation)
    """
    # Create a temporary copy of the data for calculations
    data_temp = mujoco.MjData(model)
    
    # Set joint angles
    joint_angles = np.array(joint_angles)
    if len(joint_angles) != 7:
        raise ValueError(f"Expected 7 joint angles, got {len(joint_angles)}")
    
    # Set joint positions (first 7 qpos for KUKA iiwa14)
    data_temp.qpos[:7] = joint_angles
    
    # Calculate forward kinematics
    mujoco.mj_forward(model, data_temp)
    
    # Get the site ID
    try:
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    except:
        raise ValueError(f"Site '{site_name}' not found in the model")
    
    # Get the position and orientation
    position = data_temp.site_xpos[site_id].copy()
    orientation_matrix = data_temp.site_xmat[site_id].copy().reshape(3, 3)
    
    # Convert the rotation matrix to a quaternion
    quaternion = np.zeros(4)
    mujoco.mju_mat2Quat(quaternion, orientation_matrix.flatten())
    
    return position, quaternion