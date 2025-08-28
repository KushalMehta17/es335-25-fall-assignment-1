In this question, we were asked to run our decision tree implementation on a dataset generated using the make_classification function from the sklearn.datasets module. The data was first randomly shuffled and then split into training (70%) and testing (30%) sets. We trained our decision tree on the training set and evaluated its performance on the testing set.

The dataset visualization is shown below:

![AB76D054-83A5-4B5A-A659-422854B79110](https://github.com/user-attachments/assets/56fadc6a-3108-4cbc-bd95-6885162afa95)

Running the decision tree on the dataset, we get the following results:

| Metric              | Value                 |
|---------------------|-----------------------|
| Accuracy            | 0.9666666666666667    |
| Class 0 Precision   | 0.9473684210526315    |
| Recall (Class 0)    | 1.0000                |
| Class 1 Precision   | 1.0000                |
| Recall (Class 1)    | 0.9166666666666666    |


We initially set the maximum depth of the decision tree to 5.

In the second part, we performed 5-fold nested cross-validation to determine the optimal depth of the decision tree. The cross-validation produced the following results:

| Fold | Validation Mean Depth Accuracies                         | Test Accuracy |
|------|----------------------------------------------------------|---------------|
| 0    | {3: 0.9125, 2: 0.9125, 4: 0.875, 5: 0.85, 8: 0.8375, 7: 0.8375, 6: 0.8375, 1: 0.5625} | 0.9           |
| 1    | {2: 0.9125, 4: 0.8875, 3: 0.8875, 8: 0.875, 7: 0.875, 6: 0.875, 5: 0.875, 1: 0.85}   | 0.8           |
| 2    | {2: 0.9, 3: 0.875, 6: 0.8625, 5: 0.8625, 4: 0.8625, 8: 0.85, 7: 0.85, 1: 0.525}     | 0.95          |
| 3    | {2: 0.8625, 3: 0.85, 4: 0.8125, 8: 0.775, 7: 0.775, 6: 0.775, 5: 0.775, 1: 0.525}   | 0.95          |
| 4    | {4: 0.8125, 3: 0.8125, 2: 0.8125, 6: 0.7875, 5: 0.7875, 8: 0.775, 7: 0.775, 1: 0.75} | 0.9           |
| **Mean Accuracy** | - | 0.9 |
| **Selected Depths** | [3, 2, 2, 2, 4] | - |

