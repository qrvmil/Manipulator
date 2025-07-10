from .vanilla_rrt import VanillaRRT
from utils.workspace import WorkSpace
from utils.node import Node
import numpy as np


class RRTStar(VanillaRRT):
    def __init__ (self, workspace: WorkSpace, start: tuple, goal: tuple, max_iter: int = 1000, step_size: float = 0.03, goal_radius: float = 0.5, goal_bias=0.2, collision_check_points=10, rewire_cnt=1, stop_if_reached=True, visualisation=True):
        super().__init__(workspace, start, goal, max_iter, step_size, goal_radius, goal_bias, collision_check_points, stop_if_reached, visualisation)
        self.rewire_cnt = rewire_cnt

    def dist(self, point1, point2):
        return np.linalg.norm(np.array(point1) - np.array(point2))

    def cnt_path_cost(self, node1: Node, node2: Node):
        cost = 0
        while node2 != node1:
            if node2.parent is None:
                break
            current_node = node2.parent
            cost += self.dist((node2.x, node2.y), (current_node.x, current_node.y))
            node2 = current_node
        return cost
    
    def get_nearby_nodes_with_cost(self, point):
        nearby_nodes = self.get_k_nearest_nodes(point, min(self.rewire_cnt, self.vertex_count))
        nearby_nodes_cost = [(item.object.cost + self.dist((item.object.x, item.object.y), point), item.object) for
                  item in nearby_nodes]
        return nearby_nodes_cost
    
    def rewire(self, new_node, nearby_nodes_with_cost):
        for cost, node in nearby_nodes_with_cost:
            curr_cost = node.cost
            tent_cost = new_node.cost + self.dist((new_node.x, new_node.y), (node.x, node.y))
            if tent_cost < curr_cost and self.is_collision_free((node.x, node.y), (new_node.x, new_node.y)):
                node.parent = new_node
                node.cost = tent_cost
                
    def choose_parent(self, point, nearby_nodes_with_cost):
        min_cost = float('inf')
        best_parent = None
        
        for cost, node in nearby_nodes_with_cost:
            if cost < min_cost and ((node.x, node.y) != point) and self.is_collision_free((node.x, node.y), point):
                min_cost = cost
                best_parent = node
                
        return best_parent
    
    def run_rrt_star(self):

        if self.visualisation:
            all_lines = []
            self.setup_visualization()
            self.frames.append([])

        for _ in range(self.max_iter):
            
            
            random_point = self.get_random_point()
            nearest_node = self.get_nearest_node(random_point)
            status, new_node = self.steer((nearest_node.x, nearest_node.y), random_point)

            # in case of collision
            while not status:
                random_point = self.get_random_point()
                nearest_node = self.get_nearest_node(random_point)
                status, new_node = self.steer((nearest_node.x, nearest_node.y), random_point)

            nearest_nodes_with_cost = self.get_nearby_nodes_with_cost(new_node)
            best_parent = self.choose_parent(new_node, nearest_nodes_with_cost)
            if best_parent is None:
                continue
            cost = best_parent.cost + self.dist((best_parent.x, best_parent.y), new_node)
            new_node = self.add_vertex(new_node, best_parent, cost=best_parent.cost + self.dist((best_parent.x, best_parent.y), new_node))
            new_node.cost = cost
    
            self.rewire(new_node, nearest_nodes_with_cost)
            self.completed_iterations += 1

            
            if new_node.parent and self.visualisation:
                line, = self.ax.plot([new_node.x, new_node.parent.x], 
                                [new_node.y, new_node.parent.y], "-b")
                all_lines.append(line)
                self.frames.append(all_lines.copy())
            
            # check if we reached the goal
            if np.linalg.norm([new_node.x - self.goal[0], new_node.y - self.goal[1]]) < self.goal_radius:
                if new_node.cost < self.goal_node.cost or self.goal_node.cost == 0:
                    self.goal_node = new_node
                if self.stop_if_reached:
                    break

        return self.goal_node
        