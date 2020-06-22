class Tree:
    def __init__(self, feature, value):
        self.sons = None
        self.feature = feature
        self.value = value

    def add_son(self, node):
        if self.sons is None:
            self.sons = [node]
        else:
            self.sons.append(node)

    def get_feature(self):
        return self.feature

    def get_value(self):
        return self.value
