import numpy as np
import matplotlib.pyplot as plt
import random
from rtree import index
import random
from shapely.geometry import LineString
import matplotlib.patches as patches
from utils.workspace import WorkSpace
from utils.node import Node
from utils.obstacles import RectangleObstacle, CircleObstacle, PolygonObstacle


class VanillaRRT:
    def __init__(self, workspace: WorkSpace, start: tuple, goal: tuple, max_iter: int = 1500, step_size: float = 0.03, goal_radius: float = 0.5, goal_bias=0.2, collision_check_points=10, stop_if_reached=True, visualisation=False):
        self.workspace: WorkSpace = workspace
        self.start: Node = start
        self.goal: Node = goal
        self.tree: list[Node] = [self.start]
        self.max_iter: int = max_iter
        self.completed_iterations = 0
        self.step_size: float = min(workspace.dimensions) * step_size
        self.goal_radius = self.step_size * goal_radius
        self.goal_bias = goal_bias
        self.collision_check_points = collision_check_points 
        self.stop_if_reached = stop_if_reached
        self.goal_node = Node(None, None, 0)
        self.visualisation = visualisation

        p = index.Property()
        p.dimension = len(workspace.dimensions)
        self.vertex_rtree = index.Index(interleaved=True, properties=p)
        self.vertex_count = 0
        self.head = self.add_vertex(start, None)
        

        # Параметры для графика
        if self.visualisation:
            self.fig, self.ax = plt.subplots()
            self.frames = []

    def dist(self, point1, point2):
        return np.linalg.norm(np.array(point1) - np.array(point2))

    def add_vertex(self, coords: tuple, parent: Node, cost=0):
        current = Node(coords[0], coords[1], parent=parent, cost=cost)
        if parent:
            parent.child = current
        self.vertex_count += 1
        self.vertex_rtree.insert(self.vertex_count, coords + coords, current)
        return current

    def get_nearest_node(self, point: tuple[float, float]) -> Node:
        nearest = list(self.vertex_rtree.nearest(point, 1, objects=True))[0]
        return nearest.object
        
    def get_k_nearest_nodes(self, point, k) -> list[Node]:
        k_nearest = list(self.vertex_rtree.nearest(point, k, objects=True))
        return k_nearest

    def get_random_point(self):
        goal_bias_condition = np.random.uniform(0, 1)
        if goal_bias_condition < self.goal_bias:
            x = self.goal
        else:
            x = tuple(random.uniform(0, dimension) for dimension in self.workspace.dimensions)
        return x
    
    def is_collision_free(self, start, end):
        start, end = np.array(start, dtype=np.float64), np.array(end, dtype=np.float64)
        line = LineString([start, end])
        v = end - start
        u = v / (np.sqrt(np.sum(v ** 2))) # единичный вектор
        eps = self.step_size / (self.collision_check_points + 1) # промежутков на 1 больше точек
        next_point = start
        for _ in range(self.collision_check_points + 1):
            next_point += u * eps
            point_bbox_format = np.hstack((next_point, next_point))
            if self.workspace.obstacles.count(point_bbox_format) == 0:
                continue
            else:
                possible_intersections = self.workspace.obstacles.intersection(point_bbox_format, objects=True)
                for item in possible_intersections:
                    if type(item.object) == CircleObstacle:
                        if np.linalg.norm(item.object.centers - next_point) <= item.object.radius:
                            return False
                    elif isinstance(item.object, (RectangleObstacle, PolygonObstacle)):
                        if line.intersects(item.object.shapely_polygon):
                            return False
                    # else:
                    #     return False
        return True


    def steer(self, start: tuple, end: tuple) -> tuple:
        start, end = np.array(start), np.array(end)
        v = end - start
        u = v / (np.sqrt(np.sum(v ** 2))) # единичный вектор
        steered_point = start + u * self.step_size

        if self.is_collision_free(start, steered_point):
            return True, tuple(steered_point)
        return False, None
    

    def run_rrt(self):

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

            
            new_node = self.add_vertex(new_node, nearest_node, cost=nearest_node.cost + self.dist(new_node, (nearest_node.x, nearest_node.y)))
            self.completed_iterations += 1
            
            if new_node.parent and self.visualisation:
                line, = self.ax.plot([new_node.x, new_node.parent.x], 
                                [new_node.y, new_node.parent.y], "-b")
                all_lines.append(line)
                self.frames.append(all_lines.copy())
            
            if np.linalg.norm([new_node.x - self.goal[0], new_node.y - self.goal[1]]) < self.goal_radius:
                if new_node.cost < self.goal_node.cost or self.goal_node.cost == 0:
                    self.goal_node = new_node
                if self.stop_if_reached:
                    break

        return self.goal_node

    def setup_visualization(self):
        self.ax.set_xlim(0, self.workspace.dimensions[0])
        self.ax.set_ylim(0, self.workspace.dimensions[1])
        self.ax.grid(True)

        colors = {
            RectangleObstacle: {
                "face": "lightsteelblue",
                "edge": "steelblue",
                "alpha": 0.6
            },
            CircleObstacle: {
                "face": "lightcoral",
                "edge": "indianred",
                "alpha": 0.6
            },
            PolygonObstacle: {
                "face": "lightgreen",
                "edge": "seagreen",
                "alpha": 0.6
            }
        }

        for obstacle in self.workspace.obstacles_list:
            opts = colors.get(type(obstacle), {"face": "lightgray", "edge": "gray", "alpha": 0.6})

            if isinstance(obstacle, RectangleObstacle):
                xmin, ymin, xmax, ymax = obstacle.bbox
                width, height = xmax - xmin, ymax - ymin
                patch = patches.Rectangle(
                    (xmin, ymin), width, height,
                    facecolor=opts["face"],
                    edgecolor=opts["edge"],
                    linewidth=1.5,
                    alpha=opts["alpha"]
                )

            elif isinstance(obstacle, CircleObstacle):
                patch = patches.Circle(
                    obstacle.centers, obstacle.radius,
                    facecolor=opts["face"],
                    edgecolor=opts["edge"],
                    linewidth=1.5,
                    alpha=opts["alpha"]
                )

            else:  
                patch = patches.Polygon(
                    obstacle.points, closed=True,
                    facecolor=opts["face"],
                    edgecolor=opts["edge"],
                    linewidth=1.5,
                    alpha=opts["alpha"]
                )

            self.ax.add_patch(patch)

        self.ax.plot(
            self.start[0], self.start[1],
            marker='o', markersize=10,
            color='darkblue', label='Start'
        )
        self.ax.plot(
            self.goal[0], self.goal[1],
            marker='*', markersize=12,
            color='crimson', label='Goal'
        )

        self.ax.legend(loc='upper right')
        
