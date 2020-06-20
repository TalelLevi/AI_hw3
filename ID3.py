from tree import Tree as Node
import pandas as pd
import numpy as np


class ID3:
    def __init__(self, label_index=0):
        self.label_index = label_index
        self.features = None
        self.root = None

    def train(self, df, min_examples=1):
        self.features = df.columns  # save the features names
        self.root = self._train_(df.values, min_examples)
        pass

    def _train_(self, data, min_examples):  # pruning with min_examples later
        if self.check_purity(data) or len(data) < min_examples:
            return self.classify_data(data)

        potential_splits = self.potential_splits(data)
        split_column, split_value = self.find_best_split(data, potential_splits)
        data_splits = self.split_data(data, split_column, split_value)

        node = Node(data, split_column, split_value)
        for sub_data in data_splits:
            node.add_son(self._train_(sub_data, min_examples))
        return node

    def check_purity(self, data):
        label_col = data[:, self.label_index]
        unique_classes = np.unique(label_col)
        return len(unique_classes) == 1

    def classify_data(self, data):
        label_col = data[:, self.label_index]
        unique_classifications, nr_of_counts = np.unique(label_col, return_counts=True)
        candidate_index = nr_of_counts.argmax()  # most likely answer ( if only 1 exists then this is the correct one )
        classification = unique_classifications[candidate_index]
        if len(nr_of_counts)>1 and nr_of_counts[0] == nr_of_counts[1]:
            return 1
        return unique_classifications[candidate_index]

    def potential_splits(self, data) -> dict:
        nr_of_cols = data.shape[1]  # TODO better way to get nr of cols
        potential_splits = {}  # TODO change to set in order to maintain order by col index ( feature )
        for column_index in range(nr_of_cols):
            if column_index == self.label_index:
                continue
            values = np.unique(data[:, column_index])  # get all the values for a specific column then remove duplicates
            # potential_splits[column_index] = [(values[i] + values[i + 1]) / 2 for i in range(len(values) - 1)]
            min_max_avg = (values.max() + values.min()) / 2
            potential_splits[column_index] = [min_max_avg]
        return potential_splits

    def split_data(self, data, split_column, split_value):
        values_in_split_column = data[:, split_column]
        data_under_split = data[values_in_split_column <= split_value]
        data_above_split = data[values_in_split_column > split_value]
        return data_under_split, data_above_split

    def classify(self, object_to_classify):
        if self.root is None:
            print('Decision does not exist please train first.')
        else:
            return self._classify(object_to_classify, self.root)

    def _classify(self, object_to_classify, node):
        if isinstance(node, float) or isinstance(node, int):
            return node
        feature = node.get_feature()
        objects_value_for_feature = object_to_classify[feature]
        if objects_value_for_feature <= node.get_value():
            return self._classify(object_to_classify, node.sons[0])
        else:
            return self._classify(object_to_classify, node.sons[1])

    def entropy(self, data):
        label_col = data[:, self.label_index]
        _, value_appearances_count = np.unique(label_col, return_counts=True)
        probabilities = value_appearances_count / value_appearances_count.sum()
        entropies = probabilities * -np.log2(probabilities)
        return sum(entropies)

    def information_gain(self, split_data, data):
        H = self.entropy(data)
        total_examples = len(data)
        prop_entropies = [(len(data) / total_examples) * self.entropy(data) for data in split_data]
        return H - sum(prop_entropies)

    def find_best_split(self, data, potential_splits):
        information_gain = 0
        for feature in potential_splits:
            for value in potential_splits[feature]:
                data_below, data_above = self.split_data(data, split_column=feature, split_value=value)
                current_info_gain = self.information_gain([data_below, data_above], data)

                if current_info_gain > information_gain:
                    information_gain = current_info_gain
                    best_split_feature = feature
                    best_split_value = value
        return best_split_feature, best_split_value
