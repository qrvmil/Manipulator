class Node:
    def __init__(self, q: list, cost=0, parent=None, children: list=[]):
        self.q = q
        self.parent = parent
        self.children = children
        self.cost = cost
