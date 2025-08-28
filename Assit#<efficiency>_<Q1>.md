In this task, we implemented a decision tree for the automotive efficiency dataset, limiting the tree depth to 10. The dataset was split into training and testing sets with a 70/30 ratio. Rows with missing horsepower values were removed prior to training.

We trained our decision tree on the training set and evaluated its performance on the testing set. The results are as follows:

Decision Tree (Scratch) RMSE: 6.8508

Decision Tree (Scikit-learn) RMSE: 6.6451

![09988FF9-03E5-49BB-A7D9-80B3F21D284E](https://github.com/user-attachments/assets/1a7b90fb-9216-4073-908e-e37f3168926e)


Since the RMSE values for both implementations are close, this indicates that our custom implementation is functioning correctly.

Additionally, we plotted accuracy versus tree depth for both the Scratch and Scikit-learn implementations to visualize performance trends.
