import numpy as np
import random
from rtree import index
import random
from utils.node import Node
import mujoco as mj


class VanillaRRT:
    def __init__(self, qspace: mj.Model, q_start: tuple, q_goals: list[tuple], max_iter: int = 1500,
                  step_size: float = 0.03, goal_radius: float=0.5, goal_bias=0.2,
                    sampling_frequency: int=10, stop_if_reached=True, q_limits: tuple = None):
        self.qspace: mj.Model = qspace
        self.q_start: tuple = q_start
        self.q_goals: list[tuple] = q_goals
        self.tree: list[Node] = [self.q_start]
        self.max_iter: int = max_iter
        self.completed_iterations = 0
        self.step_size: float = min(q_limits) * step_size
        self.goal_radius = self.step_size * goal_radius
        self.goal_bias = goal_bias
        self.sampling_frequency = sampling_frequency
        self.STOP_IF_REACHED = stop_if_reached

        # Parameters for goal nodes:
        self.goal_nodes = [Node(q) for q in q_goals]
        self.goal_node = self.get_nearest_goal_node(q_start)
        self.q_goal = self.goal_node.q

        # Parameters for rtree (we need rtree to find NN and KNN):
        p = index.Property()
        p.dimension = len(q_limits)
        self.vertex_rtree = index.Index(interleaved=True, properties=p)
        self.vertex_count = 0
        self.head = self.add_vertex(q_start, None)

    def get_nearest_goal_node(self, q: tuple) -> Node:
        return min(self.goal_nodes, key=lambda goal_node: self.dist(q, goal_node.q))

    def dist(self, q1, q2) -> float:
        return np.linalg.norm(np.array(q1) - np.array(q2))

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
    
    def is_collision_free_q(self, q: tuple) -> bool:
        result = True
        prev_qpos = self.qspace.data.qpos[:]
        self.qspace.data.qpos[:] = q
        self.qspace.forward()
        if self.qspace.data.ncon > 0:
            result = False
        self.qspace.data.qpos[:] = prev_qpos
        return result
    
    def is_collision_free_path(self, start, end) -> bool:
        start, end = np.array(start, dtype=np.float64), np.array(end, dtype=np.float64)
        v = end - start
        u = v / (np.sqrt(np.sum(v ** 2))) # unit vector
        eps = self.step_size / (self.sampling_frequency + 1) # amount of spaces between points = amount of points - 1
        next_point = start
        for _ in range(self.sampling_frequency + 1):
            next_point += u * eps
            if not self.is_collision_free_q(next_point):
                return False
        return True

    def steer(self, start: tuple, end: tuple) -> tuple[bool, tuple]:
        start, end = np.array(start), np.array(end)
        v = end - start
        u = v / (np.sqrt(np.sum(v ** 2))) # unit vector
        steered_q = start + u * self.step_size

        if self.is_collision_free_path(start, steered_q):
            return True, tuple(steered_q)
        return False, None
    

    def run_rrt(self):
        for _ in range(self.max_iter):
            
            random_q = self.get_random_q()
            nearest_node = self.get_nearest_node(random_q)
            status, new_q = self.steer((nearest_node.q, nearest_node.q), random_q)

            # in case of collision
            while not status:
                random_q = self.get_random_q()
                nearest_node = self.get_nearest_node(random_q)
                status, new_q = self.steer(nearest_node.q, random_q)

            new_node = self.add_vertex(new_q, nearest_node, cost=nearest_node.cost + self.dist(new_q, nearest_node.q))
            self.completed_iterations += 1
    
            for i in range(len(self.goal_nodes)):
                if np.linalg.norm([new_q - self.goal_nodes[i].q]) < self.goal_radius:
                    if new_node.cost < self.goal_nodes[i].cost or self.goal_nodes[i].cost == 0:
                        self.goal_nodes[i] = new_node
                    if self.STOP_IF_REACHED:
                        break

        return self.goal_nodes
    
    def return_path(self, goal_node: Node) -> list[tuple]:
        path = []
        while goal_node.parent:
            path.append(goal_node.q)
            goal_node = goal_node.parent
        path.append(self.q_start)
        return path[::-1]
        