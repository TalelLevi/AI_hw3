class Tree:
    def __init__(self, data, feature, value):
        self.sons = None
        self.feature = feature
        self.value = value
        self.data = data

    def add_son(self, node):
        if self.sons == None:
            self.sons = [node]
        else:
            self.sons.append(node)

    def get_feature(self):
        return self.feature

    def get_value(self):
        return self.value

    def get_data(self):
        return self.data
