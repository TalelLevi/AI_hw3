from tree import Tree as Node
import numpy as np


def euclidean_dist(vec1, vec2):
    return np.square(np.sum((vec1 - vec2) ** 2))


def get_normalization_factors(data):
    normalize_min_vals = np.array([feature.min() for feature in data.T])
    normalize_max_min_diff = np.array([feature.max() - feature.min() for feature in data.T])
    normalize_min_vals[0] = 0  # manually add 0 for label min
    normalize_max_min_diff[0] = 1  # manually add 1 for label max
    return normalize_min_vals, normalize_max_min_diff


""" K-nearest-neighbours """


class KNN:
    def __init__(self, k=9):
        self.k = k
        self.data = None
        self.labels = None
        self.normalize_min_vals = None
        self.normalize_max_min_diff = None

    def train(self, df):
        self.labels = df.values[:, :1]
        self.data = df.values[:, 1:]
        self.normalize_min_vals = np.array([feature.min() for feature in self.data.T])
        self.normalize_max_min_diff = np.array([feature.max() - feature.min() for feature in self.data.T])
        self.data = np.apply_along_axis(self.normalize, 1, self.data)

    def predict(self, examples):  # TODO support for multiple examples ?
        return self._predict(examples[1:])

    def _predict(self, example):
        normalized_example = self.normalize(example.values)
        distances = [euclidean_dist(normalized_example, known_example) for known_example in self.data]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = np.array([self.labels[index] for index in k_indices])
        labels, counts = np.unique(k_nearest_labels, return_counts=True)
        best_prediction = labels[np.argmax(counts)]
        if len(labels) > 1 and counts[0] == counts[1]:
            return True
        return best_prediction

    def normalize(self, example):
        return (example - self.normalize_min_vals) / self.normalize_max_min_diff


""" decision tree """


class ID3:
    def __init__(self, label_index=0, epsilon=False):
        self.label_index = label_index
        self.root = None
        self.epsilon = epsilon

    def train(self, df, min_examples=2):
        if self.epsilon:
            self.epsilon = [np.std(feature) * 0.1 for feature in df.values.T]
        self.root = self._train_(df.values, min_examples)

    def _train_(self, data, min_examples):  # pruning with min_examples later
        if self.check_purity(data) or len(data) < min_examples:
            return self.classify_data(data)

        potential_splits = self.potential_splits(data)
        split_column, split_value = self.find_best_split(data, potential_splits)
        data_splits = self.split_data(data, split_column, split_value)

        node = Node(split_column, split_value)
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
        classifications = {}
        for classification, appearances in zip(unique_classifications, nr_of_counts):
            classifications[classification] = appearances
        return classifications

    def potential_splits(self, data) -> dict:
        nr_of_cols = data.shape[1]
        potential_splits = {}  # TODO change to set in order to maintain order by col index ( feature )
        for column_index in range(nr_of_cols):
            if column_index == self.label_index:
                continue
            values = np.unique(data[:, column_index])  # get all the values for a specific column then remove duplicates
            potential_splits[column_index] = [(values[i] + values[i + 1]) / 2 for i in range(len(values) - 1)]
            # min_max_avg = (values.max() + values.min()) / 2
            # potential_splits[column_index] = [min_max_avg]
        return potential_splits  # dict(sorted(potential_splits.items()))

    def split_data(self, data, split_column, split_value):
        values_in_split_column = data[:, split_column]
        data_under_split = data[values_in_split_column <= split_value]
        data_above_split = data[values_in_split_column > split_value]
        return data_under_split, data_above_split

    def predict(self, object_to_classify):
        if self.root is None:
            print('Decision does not exist please train first.')
        else:
            classifications = self._predict(object_to_classify, self.root)
            if len(classifications) > 1 and classifications[0.0] == classifications[1.0]:
                return True
            return max(classifications, key=classifications.get)

    def _predict(self, object_to_classify, node):
        if isinstance(node, dict):
            return node
        feature = node.get_feature()
        objects_value_for_feature = object_to_classify[feature]
        node_split_value = node.get_value()
        if self.epsilon:
            epsilon = self.epsilon[feature]
        if self.epsilon and abs(objects_value_for_feature - node_split_value) <= epsilon:
            a = self._predict(object_to_classify, node.sons[0])
            b = self._predict(object_to_classify, node.sons[1])
            classifications = {classification: a.get(classification, 0) + b.get(classification, 0) for
                               classification in set(a) | set(b)}
            return classifications
        elif objects_value_for_feature <= node_split_value:
            return self._predict(object_to_classify, node.sons[0])
        else:
            return self._predict(object_to_classify, node.sons[1])

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
        return best_split_feature, best_split_value  # TODO info gain = 0 for everything corner case


""" epsilon decision tree with knn from neighbours """


class KnnEpsilon(ID3):
    def __init__(self, label_index=0, epsilon=False, k_value=9):
        super(KnnEpsilon, self).__init__(label_index, epsilon)
        self.normalize_min_vals = None
        self.normalize_max_min_diff = None
        self.k = k_value

    def train(self, df, min_examples=2):
        data = df.values
        self.normalize_min_vals, self.normalize_max_min_diff = get_normalization_factors(data)
        data = np.apply_along_axis(self.normalize, 1, data)
        self.epsilon = [np.std(feature) * 0.1 for feature in data.T]
        self.root = self._train_(data, min_examples)

    def classify_data(self, data):
        node = Node(0, 0, leaf=True, data=data)
        return node

    def predict(self, example):
        normalized_example = self.normalize(example.values)[1:]
        if self.root is None:
            print('Decision tree does not exist please train first.')
        else:
            node = self._predict(normalized_example, self.root)
            neighbors_to_predict_from = node.get_data()
            labels = neighbors_to_predict_from[:, :1]
            data = neighbors_to_predict_from[:, 1:]
            distances = [euclidean_dist(normalized_example, known_example) for known_example in data]
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = np.array([labels[index] for index in k_indices])
            unique_labels, counts = np.unique(k_nearest_labels, return_counts=True)
            best_prediction = unique_labels[np.argmax(counts)]
            return best_prediction

    def _predict(self, object_to_classify, node):
        if node.is_leaf():
            return node
        feature = node.get_feature()
        objects_value_for_feature = object_to_classify[feature]
        node_split_value = node.get_value()
        epsilon = self.epsilon[feature]
        if abs(objects_value_for_feature - node_split_value) <= epsilon:
            node_a = self._predict(object_to_classify, node.sons[0])
            node_b = self._predict(object_to_classify, node.sons[1])
            data_a = node_a.get_data()
            data_b = node_b.get_data()
            combined_data = np.concatenate((data_a, data_b), axis=0)
            node = Node(0, 0, leaf=True, data=combined_data)
            return node
        elif objects_value_for_feature <= node_split_value:
            node_to_return = self._predict(object_to_classify, node.sons[0])
        else:
            node_to_return = self._predict(object_to_classify, node.sons[1])
        return node_to_return

    def normalize(self, example):
        return (example - self.normalize_min_vals) / self.normalize_max_min_diff
