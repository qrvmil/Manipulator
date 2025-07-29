import sys
import os
from pathlib import Path
import time
import mujoco
import numpy as np
import uuid
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional, Union
from utils.config_loader import ConfigLoader, get_robot_config, get_simulation_config
from utils.helpers_functoins import get_robot_geom_ids, is_collision_free_q, expand_target_configs, rdp_nd, forward_kinematics

current_dir = Path(__file__).parent
sys.path.append(str(current_dir))
sys.path.append(str(current_dir.parent))

from simulator.ik.ik_simple import qpos_from_site_pose_simple
from RRT.algorithms.rrt_star import RRTStar

# 3. path post-processing
# 4. whole path control visualization


class Pipeline:

    def __init__(self, robot_name: str, config_root: Optional[str] = None):
        self.robot_name = robot_name
        self.config_loader = ConfigLoader(config_root)
        self.robot_config = self.config_loader.load_robot_config(robot_name)
        self.simulation_config = self.config_loader.load_simulation_config()

        scene_path = current_dir.parent / self.robot_config['mujoco']['default_scene']
        self.model = mujoco.MjModel.from_xml_path(str(scene_path))
        self.data = mujoco.MjData(self.model)
        self.current_targets = []
        self.target_site = self.robot_config['attachment_site']

        self.final_path_file = None

    def set_targets(self, targets: List[List[float]]):
        self.current_targets = targets
    
    def _solve_inverse_kinematics(self, target_pos: List[float]):
        scene_path = current_dir.parent / self.robot_config['mujoco']['default_scene']
        model_copy = mujoco.MjModel.from_xml_path(str(scene_path))
        data_copy = mujoco.MjData(model_copy)
        solutions = []
        original_qpos = data_copy.qpos.copy()
        for attempt in range(100):
            data_copy.qpos[:] = original_qpos[:]
        
        
            if attempt != 0:
                for i, joint_idx in enumerate(self.robot_config['joint_indices']):
                    if i < len(self.robot_config['joint_limits']):
                        low, high = self.robot_config['joint_limits'][i]
                        data_copy.qpos[joint_idx] = np.random.uniform(low, high)
                
            
            mujoco.mj_forward(model_copy, data_copy)

            if 'orientation' in self.robot_config['ik_params'].keys():
                target_quat = self.robot_config['ik_params']['orientation']
                ik_result = qpos_from_site_pose_simple(
                model=model_copy,
                data=data_copy,
                site_name=self.target_site,
                target_pos=target_pos,
                target_quat=target_quat,
                joint_indices=self.robot_config['joint_indices'],
                tol=float(self.robot_config['ik_params']['tolerance']),
                max_steps=int(self.robot_config['ik_params']['max_steps']),
                step_size=float(self.robot_config['ik_params']['step_size'])
            )
            else:
                ik_result = qpos_from_site_pose_simple(
                    model=model_copy,
                    data=data_copy,
                    site_name=self.target_site,
                    target_pos=target_pos,
                    joint_indices=self.robot_config['joint_indices'],
                    tol=float(self.robot_config['ik_params']['tolerance']),
                    max_steps=int(self.robot_config['ik_params']['max_steps']),
                    step_size=float(self.robot_config['ik_params']['step_size'])
                )

            if ik_result.success:
                is_new_solution = True
                current_config = ik_result.qpos[self.robot_config['joint_indices']]
                
                for existing_solution in solutions:
                    existing_config = existing_solution.qpos[self.robot_config['joint_indices']]
                    
                    joint_diff = np.linalg.norm(current_config - existing_config)
                    if joint_diff < 0.1:
                        is_new_solution = False
                        break
                
                if is_new_solution:
                    solutions.append(ik_result)

        self.data.qpos[:] = original_qpos[:]
        mujoco.mj_forward(self.model, self.data)
        solutions = [solution.qpos[self.robot_config['joint_indices']] for solution in solutions]
        return solutions

    def plan_one_path(self, start_pose: List[float], target_pos: List[float]):
        scene_path = current_dir.parent / self.robot_config['mujoco']['default_scene']
        model_copy = mujoco.MjModel.from_xml_path(str(scene_path))
        data_copy = mujoco.MjData(model_copy)
        
        # Устанавливаем стартовую позицию в модель
        data_copy.qpos[self.robot_config['joint_indices']] = start_pose
        mujoco.mj_forward(model_copy, data_copy)
        
        robot_geom_ids = get_robot_geom_ids(self.model, self.robot_config['robot_base'])
        solutions = self._solve_inverse_kinematics(target_pos)
        target_qs = list()

        for i, target_q in enumerate(solutions):
            goal_collision_free = is_collision_free_q(model_copy, data_copy, target_q, robot_geom_ids)
            print(f"Config #{i+1}: {'collision-free' if goal_collision_free else 'collision detected'}")
            print(f"Config #{i+1}: {target_q}")
            if goal_collision_free:
                target_qs.append(target_q)

        # target_qs = expand_target_configs(self.model, self.data, robot_geom_ids, target_qs, self.robot_config['joint_limits'])
        
        rewire_cnt = self.robot_config['default_planning'].get('rewire_count', 15)
        goal_radius = self.robot_config['default_planning'].get('goal_radius', 0.08)
        goal_bias = self.robot_config['default_planning'].get('goal_bias', 0.4)
        max_iter = self.robot_config['default_planning'].get('max_iterations', 20000)
        step_size = self.robot_config['default_planning'].get('step_size', 0.05)
        sampling_frequency = self.robot_config['default_planning'].get('sampling_frequency', 4)
        
        rrt = RRTStar(
            model_copy, 
            start_pose,  
            target_qs,
            rewire_cnt=rewire_cnt,
            q_limits=self.robot_config['joint_limits'], 
            goal_radius=goal_radius,   
            goal_bias=goal_bias,      
            max_iter=max_iter,      
            step_size=step_size,    
            sampling_frequency=sampling_frequency,  
            joint_indices=self.robot_config['joint_indices'],
            robot_geom_ids=robot_geom_ids,
            data=data_copy
        )

        goal_nodes = rrt.run_rrt_star()
        print(f"\n RRT Results:")
        print(f"   Completed iterations: {rrt.completed_iterations}")
        print(f"   Tree size: {rrt.vertex_count}")
        print(f"   Goal nodes found: {len([n for n in goal_nodes if n.cost > 0])}")

        path = []
        for goal_node in goal_nodes:
            if goal_node.cost != 0:
                path = rrt.return_path(goal_node)
                chosen_target_config = goal_node.q
                print(f'WE USE CONFIG {chosen_target_config}')
                
                target_index = -1
                min_dist = float('inf')
                for i, target_q in enumerate(target_qs):
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

        try:
            os.remove('qpath.txt')
            print("Temporary file qpath.txt deleted")
        except FileNotFoundError:
            pass
            
        return path
    

    def plan_path(self):

        robot_geom_ids = get_robot_geom_ids(self.model, self.robot_config['robot_base'])
        path = []
        for i, pos in enumerate(self.current_targets):
            if i == 0:
                new_path = self.plan_one_path(self.data.qpos[self.robot_config['joint_indices']], pos)
            else:
                new_path = self.plan_one_path(path[-1], pos)

            path += new_path

        outputs_dir = current_dir.parent / "outputs"
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_code = str(uuid.uuid4())[:8]  
        filename = f"final_path_{timestamp}_{unique_code}.txt"
        file_path = outputs_dir / filename
        
        with open(file_path, "w") as f:
            for q in path:
                f.write(" ".join(map(str, q)) + "\n")
        
        print(f"Path saved to: {file_path}")
        
        self.final_path_file = file_path
        
        return path
        


    def run_simulation(self):
            
        path_file = self.final_path_file
        path = open(path_file, 'r').readlines()
        path = [tuple(map(float, line.strip().split())) for line in path]
        # path_arrays = [np.array(point) for point in path]  
        # simplified_path = rdp_nd(path_arrays, 0.01)
        # path = [tuple(point) for point in simplified_path] 

        mujoco_settings = self.simulation_config['mujoco_settings']
        self.model.opt.timestep = mujoco_settings['timestep']
        
        solver_name = mujoco_settings['solver']
        if solver_name == "PGS":
            self.model.opt.solver = 0
        elif solver_name == "CG":
            self.model.opt.solver = 1  
        elif solver_name == "Newton":
            self.model.opt.solver = 2
        else:
            self.model.opt.solver = 0  
            
        self.model.opt.iterations = mujoco_settings['iterations']

        for i in range(min(len(self.data.ctrl), self.model.nq)):
            self.data.ctrl[i] = self.data.qpos[i]
        #mujoco.mj_forward(self.model, self.data)

        with mujoco.viewer.launch_passive(self.model, self.data, show_left_ui=False, show_right_ui=False) as viewer:
            viewer.sync()
            time.sleep(0.2)
            
            step_delay = 0.005      
            
            joint_indices = self.robot_config['joint_indices']
            
            for i, target_point in enumerate(path):
                print(f"Moving to waypoint {i+1}/{len(path)}")
                
                self.data.ctrl[joint_indices] = target_point
                
                target_tolerance = 0.05  
                max_wait_steps = 500     
                check_frequency = 10     
                
                step_count = 0
                reached_target = False
                
                while not reached_target and step_count < max_wait_steps:
                    for _ in range(check_frequency):
                        mujoco.mj_step(self.model, self.data)
                        viewer.sync()
                        time.sleep(step_delay)
                        step_count += 1
                    
                    current_position = self.data.qpos[joint_indices]
                    position_error = np.linalg.norm(np.array(target_point) - current_position)
                    
                    if position_error < target_tolerance:
                        reached_target = True
            
            try:
                while True:
                    mujoco.mj_step(self.model, self.data)
                    viewer.sync()
                    time.sleep(0.01)
            except KeyboardInterrupt:
                print("Done")
                print("FINAL CARTESIAN POSITION: ", forward_kinematics(self.model, path[-1], self.robot_config['attachment_site'])[0])
                print("FINAL JOINTS: ", self.data.qpos[:7])
                self._cleanup_temp_files()
    
    def _cleanup_temp_files(self):
        temp_files = ['qpath.txt']
        
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                pass