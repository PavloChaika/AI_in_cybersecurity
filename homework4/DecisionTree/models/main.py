import pandas as pd
import numpy as np
import graphviz
from numpy.core.defchararray import isnumeric
import torch
from models import decision_tree

malware_dataset = pd.read_csv('MalwareArtifacts.csv', delimiter=',')
test_data = pd.read_csv('test_dataset.csv', delimiter=',')
random_records = malware_dataset.sample(n=100).reset_index(drop=True)

x_list = np.array(malware_dataset.iloc[:, [0, 1, 2, 3,4,5,6]])
y_list = np.array(malware_dataset.iloc[:, -1])

x_list = np.array(test_data.iloc[:, [0, 1, 2]])
y_list = np.array(test_data.iloc[:, -1])

class Question:
    def __init__(self, variable, operation, value, impurity, number_of_children):
        self.variable = variable
        self.operation = operation
        self.value = value
        self.impurity = impurity
        self.number_of_children = number_of_children

    def __str__(self):
        return f"(x[{self.variable}] {self.operation} {self.value})"

    def match(self, x_value):
        """
        Check if the given data_point satisfies the condition defined by the question.
        """
        if self.operation == "==":
            return x_value == self.value
        elif self.operation == "<=":
            return x_value <= self.value
        else:
            raise ValueError(f"Unsupported operation: {self.operation}")


def is_numeric_column(column):
    return np.issubdtype(column.dtype, np.number)

def impurity(row):
    impurity = 1
    row_length = len(row)  # Store the length of the row in a variable
    for unique_value in set(row):
        impurity = impurity - (len([x for x in row if x == unique_value])/row_length)**2
    return impurity


def min_impurity(x, y):
    list_impurity = []
    for i in range(x.shape[1]):
        column = x[:, i]
        for unique_value in set(column):
            if is_numeric_column(column):
                operation = "<="
                condition = x[:, i] <= unique_value
                not_condition = np.logical_not(condition)
            else:
                operation = "=="
                condition = x[:, i] == unique_value
                not_condition = np.logical_not(condition)
            # Find the length of 2 splits of dataset
            len_condition = len(y[condition])
            len_not_condition = len(y[not_condition])
            # Calculate impurity for 2 splits of dataset
            true_impurity = impurity(y[condition])
            false_impurity = impurity(y[not_condition])
            # Calculate weighted impurity
            mean_impurity = len_condition / len(y) * true_impurity + len_not_condition / len(y) * false_impurity
            if len_condition == 0 or len_not_condition == 0:
                children = 1
            else:
                children = 2
            list_impurity.append(Question(i, operation, unique_value, mean_impurity, children))
    # Find the question that splits your data with the minimal impurity
    min_value = min(list_impurity, key=lambda x: x.impurity)
    return min_value


class Node:
    def __init__(self, question, parent, type = None):
        self.question = question
        self.parent = parent
        self.children = []
        self.leaf = None
        self.node_type = type

    def text_node(self):
        if self.leaf is not None:
            return f"Leaf: {self.leaf}"
        elif self.question is not None:
            return f"Node: {self.question}"
        else:
            return "Empty Node"

    def add_children(self, children):
        self.children.append(children)

    def visualize_tree(self, dot=None):
        if dot is None:
            dot = graphviz.Digraph(comment='Decision Tree')

        dot.node(str(self), self.text_node())

        for child in self.children:
            child.visualize_tree(dot)
            dot.edge(str(self), str(child), child.node_type)
        return dot

    def decision_tree(self, x,y):
        self.question = min_impurity(x,y)

        if len(set(y)) == 1:
            self.leaf = y[0]
            return

        to_be_filtered = x[:, self.question.variable]
        true_splitter = self.question.match(to_be_filtered)
        false_splitter = np.logical_not(true_splitter)

        if np.any(true_splitter):
            true_node = Node(None, self, "True")
            self.children.append(true_node)
            true_node.decision_tree(x[true_splitter], y[true_splitter])

        if np.any(false_splitter):
            false_node = Node(None, self, "False")
            self.children.append(false_node)
            false_node.decision_tree(x[false_splitter], y[false_splitter])


Start = Node("Start", None)
Start.decision_tree(x_list, y_list)
dot = Start.visualize_tree()

dot.view()