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
            # Sigmoid function output is obtained for gradient descent in Logistic Regression
            y_sigmoid = self.softmax(y_linear)
            # Get new weight and bias
            new_theta_1 = self.theta_1 - learning_rate / training_num * np.sum(y_sigmoid - y)
            new_theta_2 = self.theta_2 - learning_rate / training_num * np.dot(x.transpose(), (y_sigmoid - y))
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
        y_sigmoid = self.softmax(y_linear)
        return y_sigmoid

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



def dataset_preprocessing(csv_file="Breast_cancer_data.csv"):
    # Reading csv file
    dataset = pd.read_csv(csv_file)
    # Getting columns for x and y arrays
    # Assuming last column is output (y)
    dataset_array = dataset.to_numpy()
    np.random.shuffle(dataset_array) # Uncomment this line if shuffling operation is needed
    x, y = dataset_array[:, :-1], dataset_array[:, -1]
    test_percentage = 0.2  # It can be decreased or increased to see how accuracy percentage varies
    test_size = int(test_percentage * len(x))  # test size
    return x, y, test_size


if __name__ == "__main__":
    # Define model
    logistic_regression = MultinomialLogisticRegression()
    # Split to x, y and test_size
    x, _y, test_size = dataset_preprocessing("Multinomial_Log_Reg.csv")
    y = list()
    label_len = len(np.unique(_y))
    # One hot encoding for each array
    for k in range(len(_y)):
        yk = [0 for _ in range(label_len)]
        yk[int(_y[k])] = 1
        y.append(yk)
    y = np.array(y)
    # Split to train and test
    x_train, y_train, x_test, y_test = x[:-test_size], y[:-test_size], x[-test_size:], y[-test_size:]
    # Fit the model
    history=logistic_regression.fit(x_train, y_train, iteration_num=6000, learning_rate=1E-2,every=50)
    history_str = "\n".join(history.astype("str").tolist())
    with open("progress.txt", "w") as f:
        f.write(history_str)
    # Since 2 class is used 2 color is enough
    color_list = ["black", "red"]
    # Prediction
    y_pred = logistic_regression.prediction(x_test)
    acc = logistic_regression.accuracy(y_pred, y_test)
    print(f"Accuracy is {round(acc, 4)}")
