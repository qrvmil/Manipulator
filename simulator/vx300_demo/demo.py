import mujoco
import mujoco.viewer
import numpy as np
import time
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ik.ik_simple import qpos_from_site_pose_simple

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

def get_current_pose(model, data):
    mujoco.mj_forward(model, data)
    
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'pinch')
    
    pos = data.site_xpos[site_id].copy()
    mat = data.site_xmat[site_id].copy().reshape(3, 3)
    
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, mat.flatten())
    
    return pos, quat

def move_to_position(model, data, target_pos, target_quat=None, tolerance=1e-3):
    """
    Move the gripper to the target position using inverse kinematics
    
    Args:
        model: MuJoCo model
        data: MuJoCo data
        target_pos: target position [x, y, z]
        target_quat: target orientation [w, x, y, z] (optional)
        tolerance: acceptable position error
    
    Returns:
        bool: True if movement is successful
    """
    
    joint_indices = list(range(6))
    
    result = qpos_from_site_pose_simple(
        model=model,
        data=data,
        site_name='pinch',
        target_pos=target_pos,
        target_quat=target_quat,
        joint_indices=joint_indices,
        tol=tolerance,
        max_steps=100,
        step_size=0.05
    )
    
    if result.success:
        for i, joint_idx in enumerate(joint_indices):
            data.ctrl[joint_idx] = result.qpos[joint_idx]
        return True
    return False



def main():
    
    model = mujoco.MjModel.from_xml_path('../models/vx300/scene.xml')
    data = mujoco.MjData(model)
    
    mujoco.mj_resetDataKeyframe(model, data, 0)
    
    for i in range(min(len(data.ctrl), 7)):
        data.ctrl[i] = data.qpos[i]
    
    print_robot_state(data, "BEGIN")

    initial_pos, initial_quat = get_current_pose(model, data)
    
    phases = [
        {
            "name": "Move to cube",
            "target_pos": np.array([0.3, -0.24, 0.6]),
            "target_quat": np.array([0.7071, 0, 0, 0.7071]),  # схват вниз
            "gripper": 0.024,
            "duration": 200
        },
        {
            "name": "Open gripper",
            "target_pos": np.array([0.3, -0.24, 0.6]), 
            "target_quat": np.array([0.7071, 0, 0, 0.7071]),
            "gripper": 0.055,
            "duration": 50
        },
        {
            "name": "Lower to cube",
            "target_pos": np.array([0.3, -0.2, 0.47]), 
            "target_quat": np.array([0.7071, 0, 0, 0.7071]),
            "gripper": 0.055, 
            "duration": 150
        },
        {
            "name": "Grab cube",
            "target_pos": np.array([0.3, -0.2, 0.47]),  
            "target_quat": np.array([0.7071, 0, 0, 0.7071]),
            "gripper": 0.021,
            "duration": 50
        },
        {
            "name": "Lift with cube",
            "target_pos": np.array([0.3, -0.1, 0.7]), 
            "target_quat": np.array([0.7071, 0, 0, 0.7071]),
            "gripper": 0.021,
            "duration": 200
        },
        {
            "name": "Move cube",
            "target_pos": np.array([0.1, 0.1, 0.6]),
            "target_quat": np.array([0.7071, 0, 0, 0.7071]),
            "gripper": 0.021,
            "duration": 250
        },
        {
            "name": "Release cube",
            "target_pos": np.array([0.1, 0.1, 0.6]), 
            "target_quat": np.array([0.7071, 0, 0, 0.7071]),
            "gripper": 0.055,
            "duration": 50
        },
        {
            "name": "Move away from cube",
            "target_pos": np.array([0.1, 0.1, 0.8]),
            "target_quat": np.array([0.7071, 0, 0, 0.7071]),
            "gripper": 0.01,
            "duration": 100
        }
    ]
    
    print(f"Initial position: {initial_pos}")
    
    current_phase = 0
    phase_steps = 0
    phase_start_pos = None
    phase_start_gripper = None
    
    with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
        try:
            while current_phase < len(phases):
                phase = phases[current_phase]
                
                if phase_steps == 0:
                    current_pos, current_quat = get_current_pose(model, data)
                    phase_start_pos = current_pos
                    phase_start_gripper = data.ctrl[6]
                
                if phase_steps < phase['duration']:
                    t = phase_steps / phase['duration']
                    t_smooth = 0.5 * (1 - np.cos(np.pi * t))  # s-кривая
                    
                    interpolated_pos = phase_start_pos + t_smooth * (phase['target_pos'] - phase_start_pos)
                    interpolated_gripper = phase_start_gripper + t_smooth * (phase['gripper'] - phase_start_gripper)
                    
                    move_to_position(model, data, interpolated_pos, phase['target_quat'])
                    data.ctrl[6] = interpolated_gripper
                    
                    phase_steps += 1
                else:
                    current_phase += 1
                    phase_steps = 0
                    
                    if current_phase < len(phases):
                        print(f"Phase '{phase['name']}' completed")
                        time.sleep(0.5)
                
                mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(0.02)
            
            while True:
                mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(0.05)
                
        except KeyboardInterrupt:
            if current_phase < len(phases):
                print(f"\n last phase: {phases[current_phase]['name']}")
            print_robot_state(data, "END")
                

if __name__ == "__main__":
    main()