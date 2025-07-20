import os
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from demo.helpers_functions import *

from simulator.ik.ik_simple import qpos_from_site_pose_simple
from RRT.algorithms.vanilla_rrt import VanillaRRT
from RRT.algorithms.rrt_star import RRTStar

import time

# TODO: прописать нормально q_limits (через joint_ranges)

qlimits = [
    (-2.96706, 2.96706),
    (-2.0944, 2.0944),
    (-3.05433, 3.05433),
    (-2.0944, 2.0944),
    (-2.96706, 2.96706),
    (-2.0944, 2.0944),
    (-3.05433, 3.05433)
]


def get_path(model, data, model_copy, site_name, target_pos, joint_indices, qlimits, q_start):
    joint_indices = list(range(7))
    data_copy = mujoco.MjData(model_copy)

    ik_solutions = find_multiple_ik_solutions(
        model=model_copy,
        data=data_copy,
        site_name=site_name,
        target_pos=target_pos,
        joint_indices=joint_indices,
        num_attempts=15,
        tol=1e-6,          
        max_steps=500,     
        step_size=0.01
    )

    robot_geom_ids = get_robot_geom_ids(model, 'base')
    print("ROBOT GEOM IDS: \n", *robot_geom_ids)

    target_configs = []
    if ik_solutions:
        for i, ik_result in enumerate(ik_solutions):
            target_q = ik_result.qpos[:7]
            target_configs.append(target_q)
    else:
        ik_result = qpos_from_site_pose_simple(
            model=model_copy,
            data=data_copy,
            site_name=site_name,
            target_pos=target_pos,
            joint_indices=joint_indices,
            tol=1e-6,          
            max_steps=500,     
            step_size=0.01
        )
        target_configs = [ik_result.qpos[:7]]

    if ik_solutions:
        verify_ik_solutions(model_copy, data_copy, ik_solutions, site_name, target_pos, joint_indices)

    qlimits = [
    (-2.96706, 2.96706),
    (-2.0944, 2.0944),
    (-3.05433, 3.05433),
    (-2.0944, 2.0944),
    (-2.96706, 2.96706),
    (-2.0944, 2.0944),
    (-3.05433, 3.05433)
    ]

    rrt = RRTStar(
        model, 
        q_start, 
        target_configs,
        rewire_cnt=15,
        q_limits=qlimits, 
        goal_radius=0.08,   
        goal_bias=0.4,      
        max_iter=20000,      
        step_size=0.05,    
        sampling_frequency=4,  
        joint_indices=joint_indices,
        robot_geom_ids=robot_geom_ids,
        data=data  
    )

    start_collision_free = rrt.is_collision_free_q(data.qpos[:7])
    print(f"Start collision-free: {start_collision_free}")

    target_qs = []
    
    print(f"Target configurations collision check:")
    for i, target_q in enumerate(target_configs):
        goal_collision_free = rrt.is_collision_free_q(target_q)
        print(f"     Config #{i+1}: {'collision-free' if goal_collision_free else 'collision detected'}")
        print(f"     Config #{i+1}: {target_q}")
        if goal_collision_free:
            target_qs.append(target_q)


    target_qs = expand_target_configs(model, data, robot_geom_ids, target_qs, qlimits, noise=0.1, per_base=50)

    rrt.q_goals = target_qs
    rrt.update_goal_nodes()

    goal_nodes = rrt.run_rrt_star()
    print(f"\n RRT Results:")
    print(f"   Completed iterations: {rrt.completed_iterations}")
    print(f"   Tree size: {rrt.vertex_count}")
    print(f"   Goal nodes found: {len([n for n in goal_nodes if n.cost > 0])}")

    path = []
    chosen_target_config = None
    
    for goal_node in goal_nodes:
        if goal_node.cost != 0:
            path = rrt.return_path(goal_node)
            chosen_target_config = goal_node.q
            print(f'WE USE CONFIG {chosen_target_config}')
            
            target_index = -1
            min_dist = float('inf')
            for i, target_q in enumerate(target_configs):
                dist = np.linalg.norm(np.array(goal_node.q) - np.array(target_q))
                if dist < min_dist:
                    min_dist = dist
                    target_index = i
            
            print(f"{len(path)} points")
            print(f"target configuration #{target_index + 1}")
            print(f"target config: {[round(x, 3) for x in chosen_target_config]}")
            print(f"distance to target: {min_dist:.6f}")
            break

    path = open('qpath.txt', 'r').readlines()
    path = [tuple(map(float, line.strip().split())) for line in path]
    return path
    

def plan_final_path(model, data):
    site_name = 'attachment_site'
    model_copy = mujoco.MjModel.from_xml_path('../../simulator/models/kuka_iiwa_14/scene.xml')
    joint_indices = list(range(7))
    path1 = get_path(model, data, model_copy, site_name, [-0.5, 0.4, 0.7], joint_indices, qlimits, data.qpos[:7])
    path2 = get_path(model, data, model_copy, site_name, [-0.1, 0.4, 0.7], joint_indices, qlimits, path1[-1])
    path3 = get_path(model, data, model_copy, site_name, [0.3, 0.4, 0.7], joint_indices, qlimits, path2[-1])

    path = path1 + path2 + path3

    with open("path.txt", "w") as f:
        for q in path:
            f.write(" ".join(map(str, q)) + "\n")

    return path


def launch_simulation(model, data, joint_indices):
    # path = plan_final_path(model, data)
    path = open('final_path.txt', 'r').readlines()
    path = [tuple(map(float, line.strip().split())) for line in path]
    path_arrays = [np.array(point) for point in path]  
    simplified_path = rdp_nd(path_arrays, 0.01)
    path = [tuple(point) for point in simplified_path]  

    site_name = 'attachment_site'
    
    for i in range(min(len(data.ctrl), model.nq)):
        data.ctrl[i] = data.qpos[i]
    
    with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
        viewer.sync()
        time.sleep(0.2)  
        
        print(f"Starting trajectory with {len(path)} points...")
        
        for i, q in enumerate(path):
            print(f"\nMoving to point {i+1}/{len(path)}")
            print("TARGET JOINTS: ", [round(joint, 3) for joint in q])

            data.ctrl[joint_indices] = q
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.005) 

        final_pos = get_current_pose(model, data, site_name)[0]
        print(f"Final cartesian position: {[round(x, 3) for x in final_pos]}")

        try:
            while True:
                mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(0.005)  
        except KeyboardInterrupt:
            print_robot_state(data)
        
def main():
    model = mujoco.MjModel.from_xml_path('../../simulator/models/kuka_iiwa_14/scene.xml')
    data = mujoco.MjData(model)
    
    model.opt.timestep = 0.002 
    model.opt.solver = mujoco.mjtSolver.mjSOL_PGS
    model.opt.iterations = 50  
    
    mujoco.mj_resetDataKeyframe(model, data, 0)
    
    for i in range(min(len(data.ctrl), model.nq)):
        data.ctrl[i] = data.qpos[i]
    
    mujoco.mj_forward(model, data)
    
    joint_indices = list(range(7))
    
    launch_simulation(model, data, joint_indices)


    

def find_multiple_ik_solutions(model, data, site_name, target_pos, joint_indices, 
                               num_attempts=100, tol=1e-6, max_steps=500, step_size=0.01):
    """
    Find multiple IK solutions by trying different starting configurations
    
    Args:
        model: MuJoCo model
        data: MuJoCo data  
        site_name: name of the site to control
        target_pos: target position [x, y, z]
        joint_indices: indices of joints to optimize
        num_attempts: number of different starting configurations to try
        tol, max_steps, step_size: IK solver parameters
        
    Returns:
        List of IKResult objects with different valid solutions
    """
    solutions = []
    original_qpos = data.qpos.copy()
    
    joint_limits = [
        (-2.96706, 2.96706),  # Joint 1
        (-2.0944, 2.0944),    # Joint 2  
        (-3.05433, 3.05433),  # Joint 3
        (-2.0944, 2.0944),    # Joint 4
        (-2.96706, 2.96706),  # Joint 5
        (-2.0944, 2.0944),    # Joint 6
        (-3.05433, 3.05433)   # Joint 7
    ]
    
    print(f"   Target position: {[round(x, 3) for x in target_pos]}")
    
    for attempt in range(num_attempts):
        data.qpos[:] = original_qpos[:]
        
        if attempt == 0:
            print(f"   Attempt {attempt + 1}: Using current configuration")
        else:
            for i, joint_idx in enumerate(joint_indices):
                if i < len(joint_limits):
                    low, high = joint_limits[i]
                    data.qpos[joint_idx] = np.random.uniform(low, high)
            print(f"   Attempt {attempt + 1}: Random start {[round(data.qpos[i], 2) for i in joint_indices]}")
        
        mujoco.mj_forward(model, data)
        
        try:
            ik_result = qpos_from_site_pose_simple(
                model=model,
                data=data,
                site_name=site_name,
                target_pos=target_pos,
                target_quat=np.array([0.0,  1.0, 0.0, 0.0]),
                joint_indices=joint_indices,
                tol=tol,
                max_steps=max_steps,
                step_size=step_size
            )
            
            if ik_result.success:
                is_new_solution = True
                current_config = ik_result.qpos[joint_indices]
                
                for existing_solution in solutions:
                    existing_config = existing_solution.qpos[joint_indices]
                    
                    joint_diff = np.linalg.norm(current_config - existing_config)
                    if joint_diff < 0.1:
                        is_new_solution = False
                        break
                
                if is_new_solution:
                    solutions.append(ik_result)
            
        except Exception as e:
            print(f"IK error: {e}")

    
    
    data.qpos[:] = original_qpos[:]
    mujoco.mj_forward(model, data)
    return solutions


def verify_ik_solutions(model, data, solutions, site_name, target_pos, joint_indices):
    """
    Verify that all IK solutions actually reach the target position
    """
    original_qpos = data.qpos.copy()
    
    for i, solution in enumerate(solutions):
        for j, joint_idx in enumerate(joint_indices):
            data.qpos[joint_idx] = solution.qpos[joint_idx]
        
        mujoco.mj_forward(model, data)
        
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        current_pos = data.site_xpos[site_id].copy()
        
        pos_error = np.linalg.norm(current_pos - target_pos)
        
        print(f"Config #{i+1}: position error = {pos_error:.6f} mm")
        print(f"Current: {[round(x, 3) for x in current_pos]}")
        print(f"Target:  {[round(x, 3) for x in target_pos]}")
        
        if pos_error > 0.01:
            print(f"BAD")
        else:
            print(f"GOOD")
    
    data.qpos[:] = original_qpos[:]
    mujoco.mj_forward(model, data)


def safe_smooth_simulation(model, data, joint_indices):
    
    path = open('final_path.txt', 'r').readlines()
    path = [tuple(map(float, line.strip().split())) for line in path]

    # path_arrays = [np.array(point) for point in path]  
    # simplified_path = rdp_nd(path_arrays, 0.01)        
    # path = [tuple(point) for point in simplified_path] 

    site_name = 'attachment_site'
    
    for i in range(min(len(data.ctrl), model.nq)):
        data.ctrl[i] = data.qpos[i]
    
    with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
        viewer.sync()
        time.sleep(0.2)
        
        steps_per_point = 25      
        step_delay = 0.005       
        
        total_steps = len(path) * steps_per_point
        current_step = 0
        start_time = time.time()
        
        for i, target_point in enumerate(path):
            data.ctrl[joint_indices] = target_point
            
            for step in range(steps_per_point):
                mujoco.mj_step(model, data)
                viewer.sync()
                current_step += 1
                time.sleep(step_delay)
        
        try:
            while True:
                mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(0.01)
        except KeyboardInterrupt:
            print("Done")


def safe_smooth_main():
    model = mujoco.MjModel.from_xml_path('../../simulator/models/kuka_iiwa_14/scene.xml')
    data = mujoco.MjData(model)

    model.opt.timestep = 0.002
    model.opt.solver = mujoco.mjtSolver.mjSOL_PGS
    model.opt.iterations = 75 
    
    mujoco.mj_resetDataKeyframe(model, data, 0)
    for i in range(min(len(data.ctrl), model.nq)):
        data.ctrl[i] = data.qpos[i]
    mujoco.mj_forward(model, data)
    joint_indices = list(range(7))
    safe_smooth_simulation(model, data, joint_indices)


if __name__ == "__main__":
    safe_smooth_main()    
