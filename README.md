# PySpark-iris-flower-classification
A PySpark Application to Predict the Type of Iris Flower


A binary classification model based on Logistic Regression algorithm to predict the type of an Iris flower.

Spark Distribution: Databricks Community Edition

Data: Visit http://archive.ics.uci.edu/ml/datasets/Iris and download the dataset. To build the model, considered only 2 class attributes, i.e., we removed the records with class attribute "Iris-Setosa". Then the class attributes were dummy coded to 0s and 1s.

Model Building and Evaluation: The data was split in the ratio 0.7:0.3 to training and testing sets. Model was built on LogisticRegressionWithLBFGS algorithm and it was used to predict the values in the testing set. The performance of the model was evaluated based on the metrics: accuracy score, precision and recall.
