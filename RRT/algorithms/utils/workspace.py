import numpy as np
import random
from rtree import index
import random
from .obstacles import RectangleObstacle, CircleObstacle, PolygonObstacle


class WorkSpace:
    def __init__(self, dimensions, num_obstacles, min_obstacle_size=0.05, max_obstacle_size=0.1):
        self.dimensions = dimensions 
        self.num_obstacles: int = num_obstacles
        self.min_obstacle_size = min_obstacle_size 
        self.max_obstacle_size = max_obstacle_size 

        p = index.Property()
        p.dimension = 2
        self.obstacles = index.Index(interleaved=True, properties=p)
        self.obstacles_list = []
        self.create_obstacle_map()

        
    def create_obstacle_map(self):
        num_of_dimensions = len(self.dimensions)
        i = 0
        while i < self.num_obstacles:
            center = np.empty(num_of_dimensions, float)
            

            obstacle_type = np.random.uniform(0, 1)

            if  obstacle_type <= 0.4:
                sides_size = []
                for j, dimension in enumerate(self.dimensions):
                    min_side_length = dimension * self.min_obstacle_size
                    max_side_length = dimension * self.max_obstacle_size
                    side_size = random.uniform(min_side_length, max_side_length)
                    center[j] = random.uniform(side_size,
                                        dimension - side_size)
                    
                    sides_size.append(side_size)
                obstacle = RectangleObstacle(center, sides_size)

            elif 0.4 < obstacle_type <= 0.6:
                radius = random.uniform(min(self.dimensions) * self.min_obstacle_size, max(self.dimensions) * self.max_obstacle_size)
                for j, dimension in enumerate(self.dimensions):
                    center[j] = random.uniform(radius,
                                        dimension - radius)
                obstacle = CircleObstacle(center, radius)

            else:
                number_of_sides = random.randint(3, 7)
                points = [[0 for _ in range(num_of_dimensions)] for __ in range(number_of_sides)]
                for j, dimension in enumerate(self.dimensions):
                    min_side_length = dimension * 0.2
                    max_side_length = dimension * 0.2
                    center[j] = random.uniform(max_side_length,
                                        dimension - max_side_length)
                    for k in range(number_of_sides):
                        coord = np.random.uniform(center[j] - max_side_length, center[j] + max_side_length)
                        points[k][j] = coord
                    
                obstacle = PolygonObstacle(points)
                

            self.obstacles_list.append(obstacle)
            self.obstacles.insert(i, obstacle.bbox, obstacle)

            i += 1
