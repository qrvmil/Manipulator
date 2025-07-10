class Node:
    def __init__(self, x, y, cost=0, parent=None, child=None):
        self.x = x
        self.y = y
        self.parent = parent
        self.child = child
        self.cost = cost