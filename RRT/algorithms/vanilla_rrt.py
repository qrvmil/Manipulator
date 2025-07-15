import numpy as np
import random
from rtree import index
import random
from .utils.node import Node
import mujoco as mj
import os

#TODO: add description for class and methods
#TODO: rename file output


class VanillaRRT:
    def __init__(self, qspace: mj.MjModel, q_start: tuple, q_goals: list[tuple], joint_indices: list[int],
                 robot_geom_ids: list[int], max_iter: int=1500,
                  step_size: float = 0.03, goal_radius: float=0.5, goal_bias=0.2,
                    sampling_frequency: int=10, stop_if_reached=True, q_limits: list[tuple] = None, data: mj.MjData = None):
        self.qspace: mj.MjModel = qspace
        
        # Use provided data or create new one
        if data is not None:
            # Create a copy of the provided data to preserve world state
            self.data: mj.MjData = mj.MjData(qspace)
            # Copy the entire state from the provided data
            self.data.qpos[:] = data.qpos[:]
            self.data.qvel[:] = data.qvel[:]
            self.data.qacc[:] = data.qacc[:]
            # Forward to update all dependent quantities
            mj.mj_forward(qspace, self.data)
        else:
            self.data: mj.MjData = mj.MjData(qspace)
            
        self.q_start: tuple = q_start
        self.q_goals: list[tuple] = q_goals
        self.joint_indices: list[int] = joint_indices
        self.tree: list[Node] = [self.q_start]
        self.max_iter: int = max_iter
        self.completed_iterations = 0
        
        
        joint_ranges = [abs(limit[1] - limit[0]) for limit in q_limits]
        avg_range = np.mean(joint_ranges)
        self.step_size: float = avg_range * step_size
        
        
        self.goal_radius = goal_radius  # use absolute goal_radius in radians
        
        self.goal_bias = goal_bias
        self.sampling_frequency = sampling_frequency
        self.STOP_IF_REACHED = stop_if_reached
        self.q_limits = q_limits

        # robot geom IDs for collision filtering
        self.robot_geom_ids = robot_geom_ids

        # parameters for goal nodes:
        self.goal_nodes = [Node(q) for q in q_goals]
        self.goal_node = self.get_nearest_goal_node(q_start)
        self.q_goal = self.goal_node.q

        # parameters for rtree (we need rtree to find NN and KNN):
        p = index.Property()
        p.dimension = len(q_limits)
        self.vertex_rtree = index.Index(interleaved=True, properties=p)
        self.vertex_count = 0
        self.head = self.add_vertex(q_start, None)


    def dtheta(self, q1, q2) -> np.array:
        return np.array([(d + np.pi) % (2*np.pi) - np.pi
                        for d in (np.array(q2) - np.array(q1))])

    def get_nearest_goal_node(self, q: tuple) -> Node:
        return min(self.goal_nodes, key=lambda goal_node: self.dist(q, goal_node.q))

    def dist(self, q1, q2) -> float:
        return np.linalg.norm(self.dtheta(q1, q2))

    def add_vertex(self, q: tuple, parent: Node, cost=0) -> Node:
        current = Node(q, parent=parent, cost=cost)
        if parent:
            parent.children.append(current)
        self.vertex_count += 1
        self.vertex_rtree.insert(self.vertex_count, q + q, current)
        return current

    def get_nearest_node(self, q: tuple) -> Node:
        nearest = list(self.vertex_rtree.nearest(q, 1, objects=True))[0]
        return nearest.object
        
    def get_k_nearest_nodes(self, q, k) -> list[Node]:
        k_nearest = list(self.vertex_rtree.nearest(q, k, objects=True))
        return k_nearest

    def get_random_q(self) -> tuple:
        goal_bias_condition = np.random.uniform(0, 1)
        if goal_bias_condition < self.goal_bias:
            q = self.q_goal
        else:
            q = tuple(random.uniform(dimension[0], dimension[1]) for dimension in self.q_limits)
        return q
    
    def set_qpos(self, q: tuple) -> None:
        for i, joint_ind in enumerate(self.joint_indices):
            self.data.qpos[joint_ind] = q[i]
    

    def is_collision_free_q(self, q: tuple) -> bool:
        """Check if configuration is collision-free (only robot collisions)"""
        prev_qpos = self.data.qpos.copy()
        self.set_qpos(q)
        mj.mj_forward(self.qspace, self.data)
        
        collision_detected = False
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1_id = contact.geom1
            geom2_id = contact.geom2
            
            if geom1_id in self.robot_geom_ids or geom2_id in self.robot_geom_ids:
                collision_detected = True
                # print(f"Collision detected between geom {geom1_id} and geom {geom2_id}")
                break

        self.data.qpos[:] = prev_qpos
        mj.mj_forward(self.qspace, self.data)
        
        return not collision_detected
    
    def is_collision_free_path(self, start, end) -> bool:
        self.set_qpos(start)
        mj.mj_forward(self.qspace, self.data)
        
        
        v = self.dtheta(start, end)
        v_norm = np.sqrt(np.sum(v ** 2))
        
        # If start and end are the same, path is trivially collision-free
        if v_norm < 1e-10:
            self.set_qpos(self.q_start)
            mj.mj_forward(self.qspace, self.data)
            return True
            
        u = v / v_norm # unit vector
        eps = self.step_size / (self.sampling_frequency + 1) # amount of spaces between points = amount of points - 1
        next_point = start
        for _ in range(self.sampling_frequency + 1):
            next_point += u * eps
            if not self.is_collision_free_q(next_point):
                self.set_qpos(self.q_start)
                mj.mj_forward(self.qspace, self.data)
                return False
        self.set_qpos(self.q_start)
        mj.mj_forward(self.qspace, self.data)
        return True

    def steer(self, start: tuple, end: tuple) -> tuple[bool, tuple]:
        v = self.dtheta(start, end)
        v_norm = np.sqrt(np.sum(v ** 2))
        
        # If start and end are the same, return start position as steered point
        if v_norm < 1e-10:
            return True, start
            
        u = v / v_norm # unit vector
        steered_q = start + u * self.step_size
        steered_q = tuple(steered_q)

        if self.is_collision_free_path(start, steered_q):
            return True, tuple(steered_q)
        return False, None
    

    def run_rrt(self):
        goal_reached = False
        for _ in range(self.max_iter):
            
            random_q = self.get_random_q()
            nearest_node = self.get_nearest_node(random_q)
            status, new_q = self.steer(nearest_node.q, random_q)

            # prevent infinite loop - limit collision retry attempts
            collision_retries = 0
            max_collision_retries = 50
            
            while not status and collision_retries < max_collision_retries:
                random_q = self.get_random_q()
                nearest_node = self.get_nearest_node(random_q)
                status, new_q = self.steer(nearest_node.q, random_q)
                collision_retries += 1
            
            # skip this iteration if we couldn't find collision-free path
            if not status:
                continue

            new_node = self.add_vertex(new_q, nearest_node, cost=nearest_node.cost + self.dist(new_q, nearest_node.q))
            self.completed_iterations += 1
    
            for i in range(len(self.goal_nodes)):
                if self.dist(new_q, self.goal_nodes[i].q) < self.goal_radius:
                    if new_node.cost < self.goal_nodes[i].cost or self.goal_nodes[i].cost == 0:
                        self.goal_nodes[i] = new_node
                    if self.STOP_IF_REACHED:
                        goal_reached = True
            if goal_reached:
                break

        return self.goal_nodes
    
    def return_path(self, goal_node: Node) -> list[tuple]:
        path = []
        while goal_node.parent:
            path.append(goal_node.q)
            goal_node = goal_node.parent
        path.append(self.q_start)
        path = path[::-1]

        # Write the path to a text file in the current working directory
        current_dir = os.getcwd()
        file_path = os.path.join(current_dir, 'qpath.txt')
        with open(file_path, 'w') as file:
            for q in path:
                # Записываем только числа, разделенные пробелами
                numbers = ' '.join(str(float(x)) for x in q)
                file.write(numbers + '\n')

        return path
        