class Tree:
    def __init__(self, data):
        self.sons = None
        self.data = data

    def add_son(self, node):
        if self.sons == None:
            self.sons = [node]
        else:
            self.sons.append(node)

