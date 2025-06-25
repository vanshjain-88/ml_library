import numpy as np
import pandas as pd

def gini_impurity(y):
    if len(y) == 0:
        return 0
    probs = np.bincount(y) / len(y)
    return 1 - np.sum(probs ** 2)

def split_data(X, y, feature, threshold):
    left_mask = X[feature] <= threshold
    right_mask = ~left_mask
    return X[left_mask], y[left_mask], X[right_mask], y[right_mask]

class DecisionTreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTreeClassifier:
    def __init__(self, max_depth=None, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.root = None

    def fit(self, X, y):
        self.root = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        if len(set(y)) == 1:
            return DecisionTreeNode(value=y.iloc[0])
        
        if self.max_depth is not None and depth >= self.max_depth:
            return DecisionTreeNode(value=y.mode()[0])
        
        if len(y) < self.min_samples_leaf * 2:  # need enough samples for both children
            return DecisionTreeNode(value=y.mode()[0])

        best_gini = float('inf')
        best_feature = None
        best_threshold = None

        for feature in X.columns:
            thresholds = X[feature].unique()
            for threshold in thresholds:
                X_left, y_left, X_right, y_right = split_data(X, y, feature, threshold)

                if len(y_left) < self.min_samples_leaf or len(y_right) < self.min_samples_leaf:
                    continue

                gini_left = gini_impurity(y_left)
                gini_right = gini_impurity(y_right)

                gini_weighted = (len(y_left) * gini_left + len(y_right) * gini_right) / len(y)

                if gini_weighted < best_gini:
                    best_gini = gini_weighted
                    best_feature = feature
                    best_threshold = threshold

        if best_feature is None:
            return DecisionTreeNode(value=y.mode()[0])

        X_left, y_left, X_right, y_right = split_data(X, y, best_feature, best_threshold)
        left_subtree = self._build_tree(X_left, y_left, depth + 1)
        right_subtree = self._build_tree(X_right, y_right, depth + 1)

        return DecisionTreeNode(
            feature=best_feature,
            threshold=best_threshold,
            left=left_subtree,
            right=right_subtree
        )

    def predict(self, sample):
        return self._predict(self.root, sample)

    def _predict(self, node, sample):
        if node.value is not None:
            return node.value
        if sample[node.feature] <= node.threshold:
            return self._predict(node.left, sample)
        else:
            return self._predict(node.right, sample)

    def print_tree(self, node=None, depth=0):
        if node is None:
            node = self.root

        indent = "  " * depth
        if node.value is not None:
            print(f"{indent}Leaf: Predict = {node.value}")
        else:
            print(f"{indent}Feature '{node.feature}' <= {node.threshold}")
            self.print_tree(node.left, depth + 1)
            self.print_tree(node.right, depth + 1)
