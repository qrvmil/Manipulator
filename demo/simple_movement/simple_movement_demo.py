import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from demo.helpers_functions import *

from simulator.ik.ik_simple import qpos_from_site_pose_simple
from RRT.algorithms.vanilla_rrt import VanillaRRT
from RRT.algorithms.rrt_star import RRTStar

import time

# TODO: прописать нормально q_limits (через joint_ranges)

def find_multiple_ik_solutions(model, data, site_name, target_pos, joint_indices, 
                               num_attempts=10, tol=1e-6, max_steps=500, step_size=0.01):
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
    
    print(f"🎯 Searching for multiple IK solutions...")
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


def wait_for_position(model, data, target_q, joint_indices, tolerance=0.01, max_steps=1000, viewer=None, step_delay=0.005):
    """Wait until robot reaches target joint positions within tolerance"""
    for step in range(max_steps):
        current_q = data.qpos[joint_indices]
        distance = sum((current_q[i] - target_q[i])**2 for i in range(len(target_q)))**0.5
        if distance < tolerance:
            return True
        
        mujoco.mj_step(model, data)
        if viewer and step % 5 == 0:
            viewer.sync()
            time.sleep(step_delay)
    return False


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


def main():
    model = mujoco.MjModel.from_xml_path('../../simulator/models/kuka_iiwa_14/scene.xml')
    data = mujoco.MjData(model)
    print_robot_info(model, data)

    original_cwd = os.getcwd()
    model_dir = '../../simulator/models/kuka_iiwa_14'
    os.chdir(model_dir)
    model = mujoco.MjModel.from_xml_path('scene.xml')
    model_copy = mujoco.MjModel.from_xml_path('scene.xml')
    data_copy = mujoco.MjData(model_copy)
    
    os.chdir(original_cwd)
    
    original_timestep = model.opt.timestep
    model.opt.timestep = 0.005 
    
    # RECOMMENDATIONS FOR TIMESTEP:
    # 0.001 - Very accurate simulation (slow)
    # 0.002 - Standard accuracy (default)
    # 0.005 - Fast simulation (normal)
    # 0.01  - Very fast (may be unstable)
    # 0.02  - Extremely fast (not recommended)
    
    model.opt.solver = mujoco.mjtSolver.mjSOL_PGS
    model.opt.iterations = 100  # More iterations for stability with larger timestep
    print(f"⚙️ Solver iterations increased to {model.opt.iterations} for stability")

    for i in range(min(len(data.ctrl), model.nq)):
        data.ctrl[i] = data.qpos[i]

    site_name = 'attachment_site'

    initial_pos, initial_quat = get_current_pose(model, data, site_name)
    print(f"\nINITIAL CARTESIAN {site_name}: {[round(i, 2) for i in initial_pos]}")
    
    joint_indices = list(range(7))

    target_position = [0.08, 0.5, 0.8]
    ik_solutions = find_multiple_ik_solutions(
        model=model_copy,
        data=data_copy,
        site_name=site_name,
        target_pos=target_position,
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
            target_pos=target_position,
            joint_indices=joint_indices,
            tol=1e-6,          
            max_steps=500,     
            step_size=0.01
        )
        target_configs = [ik_result.qpos[:7]]

    if ik_solutions:
        verify_ik_solutions(model_copy, data_copy, ik_solutions, site_name, target_position, joint_indices)

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
        data.qpos[:7], 
        target_configs,
        rewire_cnt=10,
        q_limits=qlimits, 
        goal_radius=0.05,   
        goal_bias=0.5,      
        max_iter=8000,      
        step_size=0.007,    
        sampling_frequency=20,  
        joint_indices=joint_indices,
        robot_geom_ids=robot_geom_ids,
        data=data  
    )

    start_collision_free = rrt.is_collision_free_q(data.qpos[:7])
    print(f"Start collision-free: {start_collision_free}")
    
    print(f"Target configurations collision check:")
    for i, target_q in enumerate(target_configs):
        goal_collision_free = rrt.is_collision_free_q(target_q)
        print(f"     Config #{i+1}: {'collision-free' if goal_collision_free else 'collision detected'}")

    goal_nodes = rrt.run_rrt()
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

    modify_collision_parameters(model, margin=0, gap=-0.05, geom_indices=[61, 64])

    smoothed_path = rdp_nd([np.array(point) for point in path], 0.01)

    with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
        
        print("INITIAL POSITION: ", initial_pos)
        
        # VISUALIZATION DELAY SETTINGS
        trajectory_delay = 0.1  # Delay between trajectory points (in seconds)
        final_loop_delay = 0.1  # Delay in final loop
        
        # RECOMMENDATIONS FOR VISUALIZATION DELAYS:
        # trajectory_delay:
        #   0.05 - Very fast playback
        #   0.1  - Fast playback  
        #   0.2  - Normal speed (current)
        #   0.5  - Slow playback
        #   1.0  - Very slow (for detailed analysis)
        
        # final_loop_delay:
        #   0.01 - Maximum update frequency (60+ FPS)
        #   0.03 - High frequency (~30 FPS)
        #   0.07 - Standard frequency (~15 FPS)  
        #   0.1  - Low frequency (~10 FPS, current)
        
        viewer.sync()
        time.sleep(0.5)
        
        print(f"{len(path)} points...")
        for i, q in enumerate(path):
            print(f"\nMove to point {i+1}/{len(path)}")
            print("TARGET JOINTS: ", [round(joint, 3) for joint in q])

            data.ctrl[joint_indices] = q
            
            
            position_step_delay = 0.008  # Delay between steps in wait_for_position
            
            # RECOMMENDATIONS FOR position_step_delay:
            #   0.001 - Maximum smoothness (may slow down)
            #   0.005 - High smoothness
            #   0.008 - Normal smoothness (current)
            #   0.01  - Base smoothness
            #   0.02  - Minimum smoothness (faster)
            
            success = wait_for_position(model, data, q, joint_indices, 
                                      tolerance=0.02, max_steps=2000, 
                                      viewer=viewer, step_delay=position_step_delay)
            
            if success:
                current_pos = get_current_pose(model, data, site_name)[0]
                print(f"COLLISIONS: {data.ncon}")
                collisions = set(contact.geom1 for contact in data.contact) | set(contact.geom2 for contact in data.contact)
                print('GEOM COLLISIONS:', collisions)
                
            
            viewer.sync()
            time.sleep(trajectory_delay)  

        print("\nCARTESIAL FINAL POSITION: ", get_current_pose(model, data, site_name)[0])

        try:
            while True:
                mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(final_loop_delay)  
        except KeyboardInterrupt:
            print_robot_state(data)
            

if __name__ == "__main__":
    main()