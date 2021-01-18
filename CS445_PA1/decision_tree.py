"""Pure Python Decision Tree Classifier.

Simple multi-class binary decision tree classifier.
Splits are based on entropy impurity.

Initial Author: Nathan Sprague
Modified by: Anthony Thellaeche and Brockton McNerney
 molloykp -- added comments and switch impurity to entropy

"""
import numpy as np
from collections import namedtuple
import argparse

# Named tuple is a quick way to create a simple wrapper class...
Split_ = namedtuple('Split',
                    ['dim', 'pos', 'X_left', 'y_left', 'X_right', 'y_right'])


# This class does not require any student modification
# treat it as an immutable object without any methods
class Split(Split_):
    """
    Represents a possible split point during the decision tree creation process.

    Attributes:

        dim (int): the dimension along which to split
        pos (float): the position of the split
        X_left (ndarray): all X entries that are <= to the split position
        y_left (ndarray): labels corresponding to X_left
        X_right (ndarray):  all X entries that are > the split position
        y_right (ndarray): labels corresponding to X_right
    """
    pass

def split_generator(X, y):
    """
    Utility method for generating all possible splits of a data set
    for the decision tree construction algorithm.

    :param X: Numpy array with shape (num_samples, num_features)
    :param y: Numpy integer array with length num_samples
    :return: A generator for Split objects that will yield all
            possible splits of the data
    """

    # Loop over all of the dimensions.
    for dim in range(X.shape[1]):
        # Get the indices in sorted order so we can sort both  data and labels
        ind = np.argsort(X[:, dim])

        # Copy the data and the labels in sorted order
        X_sort = X[ind, :]
        y_sort = y[ind]

        # Loop through the midpoints between each point in the current dimension
        for index in range(1, X_sort.shape[0]):

            # don't try to split between equal points.
            if X_sort[index - 1, dim] != X_sort[index, dim]:
                pos = (X_sort[index - 1, dim] + X_sort[index, dim]) / 2.0

                # Yield a possible split.  Note that the slicing here does
                # not make a copy, so this should be relatively fast.
                yield Split(dim, pos,
                            X_sort[0:index, :], y_sort[0:index],
                            X_sort[index::, :], y_sort[index::])

def impurity(classes, y):
    """
    Return the impurity/entropy of the data in y

    :param y: Numpy array with class labels having shape (num_samples_in_node,)

    :return: A scalar with the entropy of the class labels
    """
    total = len(y)
    class_count = np.unique(y, return_counts=True)[1]
    return -np.sum(np.divide(class_count, total) * np.log2(np.divide(class_count, total)))

def weighted_impurity(classes, y_left, y_right):
    """
    Weighted entropy impurity for a possible split.
    :param classes: list of the classes
             y_left: class labels for the left node in the split
             y_right: class labels for the right node in the spit

    :return: A scalar with the weighted entropy
    """

    left_impurity = impurity(classes, y_left)
    right_impurity = impurity(classes, y_right)
    total_count = len(y_left) + len(y_right)
    left_count = len(y_left)
    right_count = len(y_right)
    return (left_count / total_count) * left_impurity + (right_count / total_count) * right_impurity

class DecisionTree:
    """
    A binary decision tree classifier for use with real-valued attributes.

    Attributes:
        classes (set): The set of integer classes this tree can classify.
    """

    def __init__(self, max_depth=np.inf):
        """
        Decision tree constructor.

        :param max_depth: limit on the tree depth.
                          A depth 0 tree will have no splits.
        """
        self.max_depth = max_depth
        self._root = Node()
        self.depth = 0

    def fit(self, X, y):
        """
        Construct the decision tree using the provided data and labels.

        :param X: Numpy array with shape (num_samples, num_features)
        :param y: Numpy integer array with length num_samples
        """
        self.classes = set(y)
        self.__fit_helper(self._root, X, y)

    def __fit_helper(self, node, X, y):
        parent_entropy = impurity(self.classes, y)
        if self.depth < self.max_depth and parent_entropy != 0:
            node.split = self.__get_best_split(X, y, parent_entropy)
        else:
            classes, class_count = np.unique(y, return_counts=True)
            node.class_ = classes[np.argmax(class_count)]

        if node.split is not None:
            node.left = Node()
            node.right = Node()
            self.depth += 1
            self.__fit_helper(node.left, node.split.X_left, node.split.y_left)
            self.__fit_helper(node.right, node.split.X_right, node.split.y_right)

    def __get_best_split(self, X, y, parent_entropy):
        splits = []
        for split in split_generator(X, y):
            weighted_entropy = weighted_impurity(self.classes, split.y_left, split.y_right)
            info_gain = parent_entropy - weighted_entropy
            splits.append([info_gain, split])

        splits.sort()
        return splits[-1][1]

    def predict(self, X):
        """
        Predict labels for a data set by finding the appropriate leaf node for
        each input and using the majority label as the prediction.

        :param X:  Numpy array with shape (num_samples, num_features)
        :return: A length num_samples numpy array containing predicted labels.
        """
        self.label = np.array([])
        if self.max_depth > 0:
            for x in X:
                self.__predict_helper(self._root, x)
        return self.label

    def __predict_helper(self, node, x):
        if node.split is not None:
            if x[node.split.dim] <= node.split.pos:
                self.__predict_helper(node.left, x)
            else:
                self.__predict_helper(node.right, x)
        else:
            self.label = np.append(self.label, node.class_)

    def get_depth(self):
        """
        :return: The depth of the decision tree.
        """
        return self.__get_depth_helper(self._root)

    def __get_depth_helper(self, node):
        if node is None or node.split is None:
            return 0

        left = self.__get_depth_helper(node.left)
        right = self.__get_depth_helper(node.right)

        if left > right:
            return left + 1
        else:
            return right + 1

class Node:
    """
    It will probably be useful to have a Node class.  In order to use the
    visualization code in draw_trees, the node class must have three
    attributes:

    Attributes:
        left:  A Node object or Null for leaves.
        right - A Node object or Null for leaves.
        split - A Split object representing the split at this node,
                or Null for leaves
    """
    def __init__(self, left=None, right=None, split=None, class_=None):
        self.left = left
        self.right = right
        self.split = split
        self.class_ = class_

def tree_demo():
    import draw_tree
    X = np.array([[0.88, 0.39],
                  [0.49, 0.52],
                  [0.68, 0.26],
                  [0.57, 0.51],
                  [0.61, 0.73]])
    y = np.array([1, 0, 0, 0, 1])
    tree = DecisionTree()
    tree.fit(X, y)
    draw_tree.draw_tree(X, y, tree)

def parse_args():
    parser = argparse.ArgumentParser(description='Decision Tree modeling')

    parser.add_argument('--inputFile', action='store',
                        dest='input_filename', default="", required=False,
                        help='csv data file.  Last column is the class label')

    parser.add_argument('--depthLimit', action='store', type=int,
                        dest='depth_limit', default=-1, required=False,
                        help='max depth of the decision tree')

    parser.add_argument('--testDataFile', action='store',
                        dest='test_data_filename', default="", required=False,
                        help='data file with test data')

    parser.add_argument('--treeModelFile', action='store',
                        dest='tree_model_file', default="", required=False,
                        help='output of the learned model/tree')

    parser.add_argument('-demoFlag', action='store_true',
                        dest='demo_flag',
                        help='run demo data hardcoded into program')

    return parser.parse_args()


def main():
    parms = parse_args()

    if parms.demo_flag:
        tree_demo()
    else:
        # read in training and test data
        # compute model on training data
        # run test data
        # optionally print out tree model
        pass


if __name__ == "__main__":
    main()
