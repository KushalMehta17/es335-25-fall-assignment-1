"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .utils import *

np.random.seed(42)


@dataclass
class Node:
    children: list
    criterion: str
    depth: int
    X_df: pd.DataFrame
    y_series: pd.Series
    split_feature: str
    output_type: Literal['real', 'discrete']

    def __init__(self, criterion, depth):
        self.children = []
        self.criterion = criterion
        self.depth = depth
        self.X_df = None
        self.y_series = None
        self.split_feature = None
        self.split_value = None

    def split(self, max_depth, epsilon):
        assert (self.X_df is not None) and (self.y_series is not None)

        if self.depth == max_depth:
            return

        if get_impurity_function(self.criterion)(self.y_series) <= epsilon:
            return

        # look for the attribute that yields the best split
        best_score = -np.inf
        best_value = None
        best_feature = None

        for col in self.X_df.columns:
            gain, candidate_value = information_gain(
                self.y_series, self.X_df[col], self.criterion
            )
            if gain > best_score:
                best_value = candidate_value
                best_score = gain
                best_feature = col

        if best_score == 0:
            return

        assert (best_value is not None) and (best_feature is not None)

        # set the selected split rule
        self.split_feature = best_feature
        self.split_value = best_value

        left_mask = self.X_df[best_feature] <= best_value
        right_mask = self.X_df[best_feature] > best_value

        self.children = [
            Node(self.criterion, self.depth + 1),
            Node(self.criterion, self.depth + 1),
        ]

        self.children[0].add_samples(
            self.X_df[left_mask], self.y_series[left_mask]
        )
        self.children[1].add_samples(
            self.X_df[right_mask], self.y_series[right_mask]
        )

        self.children[0].split(max_depth, epsilon)
        self.children[1].split(max_depth, epsilon)

    def add_samples(self, X_data, y_data):
        self.X_df = X_data
        self.y_series = y_data

    def get_value(self):
        if not check_ifreal(self.y_series):
            return self.y_series.mode()[0]
        else:
            return self.y_series.mean()

    def predict(self, X: pd.DataFrame):
        if len(self.children) == 0:
            return np.array(
                np.ones(X.shape[0]) * self.get_value(), dtype=self.y_series.dtype
            )

        output = np.zeros(X.shape[0], dtype=self.y_series.dtype)
        output[X[self.split_feature] <= self.split_value] = self.children[0].predict(
            X[X[self.split_feature] <= self.split_value]
        )
        output[X[self.split_feature] > self.split_value] = self.children[1].predict(
            X[X[self.split_feature] > self.split_value]
        )
        return output

    def plot(self):
        if len(self.children) == 0:
            if not check_ifreal(self.y_series):
                print(f"Class {str(self.get_value())}, {self.y_series.to_numpy()}")
            else:
                print(f"Value {str(self.get_value())}, {self.y_series.to_numpy()}")
        else:
            print(f"?{self.split_feature} <= {self.split_value}")
            print(("  " * (2 * self.depth + 1)) + "Y: ", end="")
            self.children[0].plot()
            print(("  " * (2 * self.depth + 1)) + "N: ", end="")
            self.children[1].plot()


@dataclass
class DecisionTree:
    criterion: Literal["information_gain", "gini_index"]  # criterion won't be used for regression
    max_depth: int   # The maximum depth the tree can grow to
    root: Node
    epsilon: float

    def __init__(self, criterion, max_depth=5, epsilon=1e-7):
        self.criterion = criterion
        self.max_depth = max_depth
        self.root = Node(criterion, 0)
        self.epsilon = epsilon

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Function to train and construct the decision tree
        """
        X = one_hot_encoding(X)  # convert categorical predictors
        self.root.add_samples(X, y)
        self.root.split(self.max_depth, self.epsilon)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Funtion to run the decision tree on test inputs
        """
        X = one_hot_encoding(X)
        return pd.Series(self.root.predict(X), dtype=self.root.y_series.dtype)

    def plot(self) -> None:
        """
        Show the tree splits and predictions in text format.

        Example:
        ?(X1 > 4)
            Y: ?(X2 > 7)
                Y: Class A
                N: Class B
            N: Class C
        """
        self.root.plot()
