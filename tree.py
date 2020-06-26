class Tree:
    def __init__(self, feature, value, leaf=False, data=None):
        self.sons = None
        self.feature = feature
        self.value = value

        self.leaf = leaf
        self.data = data

    def add_son(self, node):
        if self.sons is None:
            self.sons = [node]
        else:
            self.sons.append(node)

    def get_feature(self):
        return self.feature

    def get_value(self):
        return self.value

    def is_leaf(self):
        return self.leaf

    def get_data(self):
        return self.data