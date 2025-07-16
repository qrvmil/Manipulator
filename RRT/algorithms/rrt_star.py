from .vanilla_rrt import VanillaRRT
from .utils.node import Node
import numpy as np
import mujoco as mj

# NB! STOP_IF_REACHED should be false for RRT*, because we want to find the best path, not the first one


class RRTStar(VanillaRRT):
    def __init__ (self, qspace: mj.MjModel, q_start: tuple, q_goals: list[tuple], joint_indices: list[int],
                 robot_geom_ids: list[int], rewire_cnt: int=1, max_iter: int=1500,
                  step_size: float = 0.03, goal_radius: float=0.5, goal_bias=0.2,
                    sampling_frequency: int=10, stop_if_reached=True, q_limits: list[tuple] = None, data: mj.MjData = None):
        super().__init__(qspace, q_start, q_goals, joint_indices, robot_geom_ids, max_iter, step_size, goal_radius, goal_bias, sampling_frequency, stop_if_reached, q_limits, data)
        self.rewire_cnt = rewire_cnt

    def cnt_path_cost(self, node1: Node, node2: Node) -> float:
        cost = 0
        while node2 != node1:
            if node2.parent is None:
                break
            current_node = node2.parent
            cost += self.dist(node2.q, current_node.q)
            node2 = current_node
        return cost
    
    def get_nearby_nodes_with_cost(self, q: tuple) -> list[tuple[float, Node]]:
        nearby_nodes = self.get_k_nearest_nodes(q, min(self.rewire_cnt, self.vertex_count))
        nearby_nodes_cost = [(item.object.cost + self.dist(item.object.q, q), item.object) for
                  item in nearby_nodes]
        return nearby_nodes_cost
    
    def rewire(self, new_node, nearby_nodes_with_cost):
        for cost, node in nearby_nodes_with_cost:
            curr_cost = node.cost
            tent_cost = new_node.cost + self.dist(new_node.q, node.q)
            if tent_cost < curr_cost and self.is_collision_free_path(new_node.q, node.q):
                node.parent = new_node
                node.cost = tent_cost
                
    def choose_parent(self, q, nearby_nodes_with_cost):
        min_cost = float('inf')
        best_parent = None
        
        for cost, node in nearby_nodes_with_cost:
            if cost < min_cost and (node.q != q) and self.is_collision_free_path(node.q, q):
                min_cost = cost
                best_parent = node
                
        return best_parent
    
    def run_rrt_star(self):
        goal_reached = False
        for _ in range(self.max_iter):
            
            random_q = self.get_random_q()
            nearest_node = self.get_nearest_node(random_q)
            status, new_node = self.steer(nearest_node.q, random_q)

            # in case of collision
            while not status:
                random_q = self.get_random_q()
                nearest_node = self.get_nearest_node(random_q)
                status, new_node = self.steer(nearest_node.q, random_q)

            nearest_nodes_with_cost = self.get_nearby_nodes_with_cost(new_node)
            best_parent = self.choose_parent(new_node, nearest_nodes_with_cost)
            if best_parent is None:
                continue
            cost = best_parent.cost + self.dist((best_parent.x, best_parent.y), new_node)
            new_node = self.add_vertex(new_node, best_parent, cost=best_parent.cost + self.dist((best_parent.x, best_parent.y), new_node))
            new_node.cost = cost
    
            self.rewire(new_node, nearest_nodes_with_cost)
            self.completed_iterations += 1

            for i in range(len(self.goal_nodes)):
                if self.dist(new_node.q, self.goal_nodes[i].q) < self.goal_radius:
                    if new_node.cost < self.goal_nodes[i].cost or self.goal_nodes[i].cost == 0:
                        self.goal_nodes[i] = new_node
                    if self.STOP_IF_REACHED:
                        goal_reached = True
            if goal_reached:
                break

        return self.goal_nodes
    

        