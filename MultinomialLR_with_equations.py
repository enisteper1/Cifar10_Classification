import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class MultinomialLogisticRegression:

    def __init__(self):
        self.theta_1 = 0  # Bias
        self.theta_2 = None  # Weights

    def fit(self, x, y, iteration_num=8000, learning_rate=1E-2, every=10):
        np.seterr(divide="ignore", invalid="ignore", over="ignore")
        accuracy_list = list()
        # Get samples
        training_num = x.shape[0]
        # Initialize weights and bias
        if self.theta_1 == 0:
            self.theta_1 = 1
            self.theta_2 = np.ones((x.shape[1], y.shape[1]))
        # Train respect to iteration num
        for i in range(iteration_num):
            # In order to make prediction about logistic reg. linear model is calculated
            y_linear = self.theta_1 + np.dot(x, self.theta_2)
            # Softmax function output is obtained for gradient descent in Logistic Regression
            y_softmax = self.softmax(y_linear)
            # Get new weight and bias
            new_theta_1 = self.theta_1 - learning_rate / training_num * np.sum(y_softmax - y)
            new_theta_2 = self.theta_2 - learning_rate / training_num * np.dot(x.transpose(), (y_softmax - y))
            # Assign the new weight and bias
            self.theta_1 = new_theta_1
            self.theta_2 = new_theta_2
            if i % every == 0:
                pred = self.prediction(x)
                accuracy = self.accuracy(pred, y)
                accuracy_list.append(accuracy)
            if i % 500 == 0:
                print(f"Epoch: {i}")
        return np.array(accuracy_list)

    # Prediction
    def prediction(self, x):
        # Linear model
        y_linear = self.theta_1 + np.dot(x, self.theta_2)
        # Logistic Regression outputs
        y_softmax = self.softmax(y_linear)
        return y_softmax

    # softmax function
    def softmax(self, y_linear):
        epsilon = 1E-5
        try:
            y_soft = (np.exp(y_linear.T) / np.sum(np.exp(y_linear), axis=1)).T
        except Exception as ex:
            print(ex)
            y_soft = np.ones(y_linear.shape[1])
        return y_soft

    # Accuracy calculation
    def accuracy(self, output_predicted, output_true):
        true_counter = 0
        for i in range(len(output_true)):
            if np.argmax(output_true[i]) == np.argmax(output_predicted[i]):
                true_counter += 1
        # Get comparison of predicted and true output
        acc = true_counter / len(output_true)
        return acc