import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from sklearn.datasets import make_classification

np.random.seed(42)

# Code given in the question
X, y = make_classification(
    n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5)

# Shuffle X and y
shuffle_idx = np.random.permutation(y.size)
X, y = X[shuffle_idx], y[shuffle_idx]

def print_report(y, y_pred):
    print("Accuracy: ", accuracy(y_pred, y))
    for cls in y.unique():
        print("Class: ", cls)
        print("Precision: ", precision(y_pred, y, cls))
        print("Recall: ", recall(y_pred, y, cls))

def prepare_dataset():
    total_size = len(X)
    train_size = int(total_size * 0.7)

    X_train = pd.DataFrame(X[:train_size])
    y_train = pd.Series(y[:train_size])
    X_test = pd.DataFrame(X[train_size:])
    y_test = pd.Series(y[train_size:])

    return X_train, y_train, X_test, y_test


# For plotting
scatter = plt.scatter(X[:, 0], X[:, 1], c=y)
plt.title("Dataset")
plt.xlabel("First feature")
plt.ylabel("Second feature")
plt.gca().add_artist(plt.legend(*scatter.legend_elements(), loc="upper right", title="Labels"))
plt.show()

# Q2a
def evaluate_tree():
    X_train, y_train, X_test, y_test = prepare_dataset()
    tree = DecisionTree(criterion='information_gain')
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)
    print_report(y_test, y_pred)


# Q2b
# Use 5 fold cross-validation on the dataset. Using nested cross-validation find the optimum depth of the tree. [1 mark] Implement cross-validation from scratch.
def inner_cross_validation(X, y, n_folds, depths):
    fold_size = len(X) // n_folds
    d_acc = {}
    for depth in depths:
        acc = []
        for i in range(n_folds):
            X_train = pd.concat([X[:i * fold_size], X[(i + 1) * fold_size:]])
            y_train = pd.concat([y[:i * fold_size], y[(i + 1) * fold_size:]])
            X_val = X[i * fold_size:(i + 1) * fold_size]
            y_val = y[i * fold_size:(i + 1) * fold_size]
            
            tree = DecisionTree(criterion='information_gain', max_depth=depth)
            tree.fit(X_train, y_train)
            y_pred = tree.predict(X_val)
            acc.append(accuracy(y_pred.to_numpy(), y_val.to_numpy()))

        d_acc[depth] = np.mean(acc).item()
    d_acc = dict(sorted(d_acc.items(), key=lambda item: (item[1], item[0]), reverse=True))
    return d_acc

def outer_cross_validation(X, y, n_folds, depths):
    fold_size = len(X) // n_folds
    acc = []
    depths_per_fold = []
    
    for i in range(n_folds):
        X_train = pd.concat([X[:i * fold_size], X[(i + 1) * fold_size:]])
        y_train = pd.concat([y[:i * fold_size], y[(i + 1) * fold_size:]])
        X_test = X[i * fold_size:(i + 1) * fold_size]
        y_test = y[i * fold_size:(i + 1) * fold_size]
        
        depths_acc = inner_cross_validation(X_train, y_train, 5, depths)
        best_depth = list(depths_acc.keys())[0]
        tree = DecisionTree(criterion='information_gain', max_depth=best_depth)
        tree.fit(X_train, y_train)
        y_pred = tree.predict(X_test)
        acc.append(accuracy(y_pred.to_numpy(), y_test.to_numpy()))
        depths_per_fold.append(best_depth)
        print(f"Fold {i}, Mean Depth Accuracies on Validation {depths_acc}")
        print (f"Fold {i}, Accuracy on Test dataset: {acc[-1]}")
        
    return np.mean(acc), depths_per_fold

def evaluate_k_fold_nested_cross_validation(X, y, n_folds, depths):
    acc, depths = outer_cross_validation(X, y, n_folds, depths)
    print(f"Mean Accuracy accross {n_folds} folds: ", acc)
    print("Depths: ", depths)

evaluate_tree()
evaluate_k_fold_nested_cross_validation(pd.DataFrame(X), pd.Series(y), 5, [1, 2, 3, 4, 5, 6, 7, 8])
