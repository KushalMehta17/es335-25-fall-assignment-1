import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from tqdm import tqdm

np.random.seed(42)
num_average_time = 100 # Number of times to run each experiment to calculate the average values

# Train time should be O(N * (2 ^ d) * M)
# Testing time should be O(d * N)

# Function to create fake data (take inspiration from usage.py)
# Real input real output
N_values = [10, 30, 50, 70, 100]
M_values = [5, 10, 15, 20, 25]

def get_data(type, N, M):
    np.random.seed(42)
    
    if type == 'real_input_real_output':
        X = pd.DataFrame(np.random.randn(N, M))
        y = pd.Series(np.random.randn(N))
        return X, y
    elif type == 'real_input_discrete_output':
        X = pd.DataFrame(np.random.randn(N, M))
        y = pd.Series(np.random.randint(0, 2, N))
        return X, y
    elif type == 'discrete_input_real_output':
        X = pd.DataFrame(np.random.randint(0, 2, (N, M)))
        y = pd.Series(np.random.randn(N))
        return X, y
    elif type == 'discrete_input_discrete_output':
        X = pd.DataFrame(np.random.randint(0, 2, (N, M)))
        y = pd.Series(np.random.randint(0, 2, N))
        return X, y
    else:
        raise ValueError("Invalid type")

def get_decision_tree_time(N, M, type):
    X, y = get_data(type, N, M)
    training_times = []
    testing_times = []
    for _ in tqdm(range(num_average_time)):
        tree = DecisionTree(criterion='information_gain', max_depth=10)
        start = time.process_time()
        tree.fit(X, y)
        end = time.process_time()
        training_times.append(end - start)
        start = time.process_time()
        tree.predict(X)
        end = time.process_time()
        testing_times.append(end - start)
    
    print(f"For N={N}, M={M} Average training time: {np.mean(training_times)}" + f" Average testing time: {np.mean(testing_times)}")
    
    return np.mean(training_times), np.mean(testing_times), np.std(training_times), np.std(testing_times)

def plot_twin_axis_graph(x, y1, y2, y1_std, y2_std, title, xlabel):
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:red'
    ax1.set_title(title + " Training")
    ax1.set_ylabel('time')
    ax1.set_xlabel(xlabel, color=color)
    ax1.plot(x, y1, color=color)
    ax1.errorbar(x, y1, yerr=y1_std, fmt='o', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(["Training Time", "Training time $\pm$ 1$\sigma$"]) 
    plt.show()

    fig, ax1 = plt.subplots(figsize=(10, 6))
    color = 'tab:blue'
    ax1.set_title(title + " Testing")
    ax1.set_ylabel('time')
    ax1.set_xlabel(xlabel, color=color)
    ax1.plot(x, y2, color=color)
    ax1.errorbar(x, y2, yerr=y2_std, fmt='o', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(["Testing Time", "Testing time $\pm$ 1$\sigma$"]) 
    plt.show()


def plot_graph(N_vals, M_vals, fn, type):
    print("Plotting graph for", type)
    plt.title(type + " wrt N")
    plt.xlabel('N')
    plt.ylabel('Time')

    training_times = []
    testing_times = []
    training_time_stds = []
    testing_time_stds = []

    for i in N_vals:
        training_time, testing_time, training_time_std, testing_time_std = fn(i, 5, type)
        training_times.append(training_time)
        testing_times.append(testing_time)
        training_time_stds.append(training_time_std)
        testing_time_stds.append(testing_time_std)
        
    # Plot the graph in twin axis
    plot_twin_axis_graph(N_vals, training_times, testing_times, training_time_stds, testing_time_stds, type + " wrt N", 'N')

    training_times = []
    testing_times = []
    testing_time_stds = []
    training_time_stds = []
    
    for i in M_vals:
        training_time, testing_time, training_time_std, testing_time_std = fn(20, i, type)
        training_times.append(training_time)
        testing_times.append(testing_time)
        training_time_stds.append(training_time_std)
        testing_time_stds.append(testing_time_std)
        
    plot_twin_axis_graph(M_vals, training_times, testing_times, training_time_stds, testing_time_stds, type + " wrt M", 'M')
    
plot_graph(N_values, M_values, get_decision_tree_time, "real_input_real_output")
plot_graph(N_values, M_values, get_decision_tree_time, "real_input_discrete_output")
plot_graph(N_values, M_values, get_decision_tree_time, "discrete_input_real_output")
plot_graph(N_values, M_values, get_decision_tree_time, "discrete_input_discrete_output")
# ...
# Function to calculate average time (and std) taken by fit() and predict() for different N and P for 4 different cases of DTs
# ...
# Function to plot the results
# ...
# Other functions
# ...
# Run the functions, Learn the DTs and Show the results/plots
