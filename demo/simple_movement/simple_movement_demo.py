import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from demo.helpers_functions import *

from simulator.ik.ik_simple import qpos_from_site_pose_simple
from RRT.algorithms.vanilla_rrt import VanillaRRT
from RRT.algorithms.rrt_star import RRTStar

import time


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

    ik_result = qpos_from_site_pose_simple(
        model=model_copy,
        data=data_copy,
        site_name=site_name,
        target_pos=[0.08, 0.5, 0.8],
        joint_indices=joint_indices,
        tol=1e-6,          
        max_steps=500,     
        step_size=0.01
    )

    robot_geom_ids = get_robot_geom_ids(model, 'base')
    print("ROBOT GEOM IDS: \n", *robot_geom_ids)

    target_q = ik_result.qpos[:7]
    target_pos = data.site_xpos[0]

    print("\nINITIAL JOINTS (robot only): ", data.qpos[:7])
    print("TARGET JOINTS (robot only): ", target_q)

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
        [target_q], 
        rewire_cnt=5,
        q_limits=qlimits, 
        goal_radius=0.05,   
        goal_bias=0.5,      
        max_iter=8000,      
        step_size=0.007 ,    
        sampling_frequency=20,  
        joint_indices=joint_indices,
        robot_geom_ids=robot_geom_ids,
        data=data  
    )

    start_collision_free = rrt.is_collision_free_q(data.qpos[:7])
    goal_collision_free = rrt.is_collision_free_q(target_q)
    print(f"   Start collision-free: {start_collision_free}")
    print(f"   Goal collision-free: {goal_collision_free}")

    goal_nodes = rrt.run_rrt()
    print(f"\n RRT Results:")
    print(f"   Completed iterations: {rrt.completed_iterations}")
    print(f"   Tree size: {rrt.vertex_count}")
    print(f"   Goal nodes found: {len([n for n in goal_nodes if n.cost > 0])}")

    path = []
    
    for goal_node in goal_nodes:
        nearest = rrt.get_nearest_node(goal_node.q)
        print(nearest.q, "DIST: ", rrt.dist(nearest.q, goal_node.q))
        print(rrt.is_collision_free_path(nearest.q, goal_node.q))
        print()
        if goal_node.cost != 0:
            path = rrt.return_path(goal_node)
            print(f"Found path with {len(path)} points")
            break

    path = open('qpath.txt', 'r').readlines()
    path = [tuple(map(float, line.strip().split())) for line in path]

    modify_collision_parameters(model, margin=0, gap=-0.05, geom_indices=[61, 64])
    

    print(model.geom_margin[61])
    print(model.geom_gap[61])

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