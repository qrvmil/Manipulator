import numpy as np
import mujoco
from collections import deque

import numpy as np


def modify_collision_parameters(model, margin=None, gap=None, geom_indices=None):
    """
    Modify collision detection parameters for specified geometries or all geometries
    
    Args:
        model: MuJoCo model
        margin: New margin value (if None, keep original)
        gap: New gap value (if None, keep original)  
        geom_names: List of geometry names to modify (if None, modify all)
    """
    
    if geom_indices is None:
        geom_indices = range(model.ngeom)
        print(f"Changing collision parameters for all {model.ngeom} geometries")
    
    for geom_id in geom_indices:
        original_margin = model.geom_margin[geom_id]
        original_gap = model.geom_gap[geom_id]
        
        if margin is not None:
            model.geom_margin[geom_id] = margin
        if gap is not None:
            model.geom_gap[geom_id] = gap
            
        print(f"  Geometry {geom_id}: margin {original_margin:.3f}→{model.geom_margin[geom_id]:.3f}, "
              f"gap {original_gap:.3f}→{model.geom_gap[geom_id]:.3f}")
    


def is_collision_free_q(model, data, q, robot_geom_ids):
    prev_qpos = data.qpos.copy()
    data.qpos[:7] = q
    mujoco.mj_forward(model, data)
    
    collision_detected = False
    for i in range(data.ncon):
        contact = data.contact[i]
        geom1_id = contact.geom1
        geom2_id = contact.geom2
        
        if geom1_id in robot_geom_ids or geom2_id in robot_geom_ids:
            collision_detected = True
            # print(f"Collision detected between geom {geom1_id} and geom {geom2_id}")
            break

    data.qpos[:] = prev_qpos
    mujoco.mj_forward(model, data)
    
    return not collision_detected

def expand_target_configs(model, data, robot_geom_ids, base_configs, 
                          q_limits, 
                          noise=0.1,   # радиус гауссовского шума, рад
                          per_base=50  # сколько новых конфигов на каждую базовую
                         ):
    """
    Создаём «облако» валидных целевых конфигураций вокруг уже найденных IK-решений.
    rrt         – готовый экземпляр RRTStar (должен уметь проверять коллизии)
    base_configs – список существующих q (len == nq)
    q_limits    – список (min,max) на сустав
    """
    all_targets = list(base_configs)
    for q_base in base_configs:
        for _ in range(per_base):
            q_new = np.array(q_base) + np.random.normal(0, noise, size=len(q_base))
            for i, (lo, hi) in enumerate(q_limits):
                q_new[i] = np.clip(q_new[i], lo, hi)
            if is_collision_free_q(model, data, q_new, robot_geom_ids):
                all_targets.append(tuple(q_new))
    return all_targets

def perpendicular_distance(pt, start, end):
    v = end - start
    if np.allclose(v, 0):
        return np.linalg.norm(pt - start)
    t = np.dot(pt - start, v) / np.dot(v, v)
    t = np.clip(t, 0.0, 1.0)
    proj = start + t * v
    return np.linalg.norm(pt - proj)

def rdp_nd(points, epsilon):
    if len(points) < 3:
        return points

    start, end = points[0], points[-1]
    dists = [perpendicular_distance(pt, start, end) for pt in points[1:-1]]
    idx_max = int(np.argmax(dists)) + 1
    if dists[idx_max-1] > epsilon:
        left  = rdp_nd(points[:idx_max+1], epsilon)
        right = rdp_nd(points[idx_max:], epsilon)
        return left[:-1] + right
    else:
        return [start, end]



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