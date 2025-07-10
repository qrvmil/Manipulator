import numpy as np
from shapely.geometry import Polygon


class RectangleObstacle:
    def __init__(self, centers, side_sizes):
        self.centers = centers
        self.side_sizes = side_sizes
        self.bbox = self.get_bbox()
        self.shapely_polygon = self.get_shapely_polygon()

    def get_shapely_polygon(self):
        xmin = self.centers[0] - self.side_sizes[0]
        ymin = self.centers[1] - self.side_sizes[1]
        xmax = self.centers[0] + self.side_sizes[0]
        ymax = self.centers[1] + self.side_sizes[1]
        
        points = [
            (xmin, ymin),
            (xmax, ymin),
            (xmax, ymax),
            (xmin, ymax)
        ]
        return Polygon(points)

    
    def get_bbox(self):
        num_of_dimensions = len(self.centers)
        min_corner = np.empty(num_of_dimensions, float)
        max_corner = np.empty(num_of_dimensions, float)
        for j in range(num_of_dimensions):
            min_corner[j] = self.centers[j] - self.side_sizes[j]
            max_corner[j] = self.centers[j] + self.side_sizes[j]
        bbox = np.append(min_corner, max_corner)
        return tuple(bbox)
        
class PolygonObstacle:
    def __init__(self, points):
        self.points = points
        self.shapely_polygon = Polygon(points)
        self.bbox = self.get_bbox()

    def get_bbox(self):
        num_of_dimensions = len(self.points[0])
        min_corner = np.empty(num_of_dimensions, float)
        max_corner = np.empty(num_of_dimensions, float)
        for j in range(num_of_dimensions):
            min_corner[j] = min([i[j] for i in self.points])
            max_corner[j] = max([i[j] for i in self.points])
        bbox = np.append(min_corner, max_corner)
        return tuple(bbox)

class CircleObstacle:
    def __init__(self, centers, radius):
        self.centers = centers
        self.radius = radius
        self.bbox = self.get_bbox()

    def get_bbox(self):
        num_of_dimensions = len(self.centers)
        min_corner = np.empty(num_of_dimensions, float)
        max_corner = np.empty(num_of_dimensions, float)
        for j in range(num_of_dimensions):
            min_corner[j] = self.centers[j] - self.radius
            max_corner[j] = self.centers[j] + self.radius
        bbox = np.append(min_corner, max_corner)
        return tuple(bbox)
    
