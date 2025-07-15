from helpers_functions import *
import os
import sys
import mujoco

# Add the parent directory to sys.path to enable absolute imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from simulator.ik.ik_simple import qpos_from_site_pose_simple
from RRT.algorithms.vanilla_rrt import VanillaRRT
import time


def wait_for_position(model, data, target_q, joint_indices, tolerance=0.01, max_steps=1000, viewer=None):
    """Wait until robot reaches target joint positions within tolerance"""
    for step in range(max_steps):
        # Get current joint positions
        current_q = data.qpos[joint_indices]
        
        # Calculate distance to target
        distance = sum((current_q[i] - target_q[i])**2 for i in range(len(target_q)))**0.5
        
        if distance < tolerance:
            return True
            
        # Continue simulation
        mujoco.mj_step(model, data)
        
        # Update viewer for smooth visualization
        if viewer and step % 5 == 0:  # Update viewer every 5 steps
            viewer.sync()
            time.sleep(0.002)  # Small delay for real-time visualization
    
    return False  # Timeout


def main():
    model = mujoco.MjModel.from_xml_path('../simulator/models/kuka_iiwa_14/scene.xml')
    data = mujoco.MjData(model)
    print_robot_info(model, data)

    original_cwd = os.getcwd()
    model_dir = '../simulator/models/kuka_iiwa_14'
    os.chdir(model_dir)
    model = mujoco.MjModel.from_xml_path('scene.xml')
    model_copy = mujoco.MjModel.from_xml_path('scene.xml')
    data_copy = mujoco.MjData(model_copy)
    
    # Return to original directory after loading models
    os.chdir(original_cwd)

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
        target_pos=[0.1, 0.5, 0.8],
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

    rrt = VanillaRRT(
        model, 
        data.qpos[:7], 
        [target_q], 
        q_limits=qlimits, 
        goal_radius=0.05,   
        goal_bias=0.3,      
        max_iter=8000,      
        step_size=0.01,    
        sampling_frequency=50,  
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
            print(f"Найден путь с {len(path)} точками")
            break

    path = open('qpath.txt', 'r').readlines()
    path = [tuple(map(float, line.strip().split())) for line in path]

    with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
        
        print("INITIAL POSITION: ", initial_pos)
        # Removed 100 simulation steps that were changing world state
        
        # Small pause for viewer synchronization
        viewer.sync()
        time.sleep(0.5)
        
        print(f"{len(path)} точек...")
        for i, q in enumerate(path):
            print(f"\nMove to point {i+1}/{len(path)}")
            print("TARGET JOINTS: ", [round(joint, 3) for joint in q])

            data.qpos[:7] = q
            mujoco.mj_forward(model, data)
            collisions = set(contact.geom1 for contact in data.contact) | set(contact.geom2 for contact in data.contact)
            print('GEOM COLLISIONS:', collisions)
            
            # data.ctrl[joint_indices] = q
            
            # success = wait_for_position(model, data, q, joint_indices, tolerance=0.02, max_steps=2000, viewer=viewer)
            
            # if success:
            #     current_pos = get_current_pose(model, data, site_name)[0]
            #     print(f"COLLISIONS: {data.ncon}")
            #     collisions = set(contact.geom1 for contact in data.contact) | set(contact.geom2 for contact in data.contact)
            #     print('GEOM COLLISIONS:', collisions)
                
            
            viewer.sync()
            time.sleep(0.1)  

        print("\nCARTESIAL FINAL POSITION: ", get_current_pose(model, data, site_name)[0])


        try:
            while True:
                viewer.sync()
                time.sleep(0.07)
        except KeyboardInterrupt:
            print_robot_state(data)
            





if __name__ == "__main__":
    main()