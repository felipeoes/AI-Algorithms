import pandas as pd


class Perceptron(object):
    def __init__(self, weights: dict, bias: float):
        self.weights = weights
        self.bias = bias

    def train(self, X: list, y: list, epochs: int = 10, learning_rate: float = 0.1):
        """ Trains the perceptron according to specified parameters ."""

        # alterando o problema especificamente para usar a função step bipolar
        y = [-1 if num == 0 else num for num in y]

        accuracy_list = []

        for _ in range(epochs):
            last_weights = self.weights.copy()
            last_bias = self.bias
            y_true = 0
            y_false = 0

            for x, y_ in zip(X, y):
                y_in = self.bias + \
                    sum([x[col_num] * self.weights[col_num]
                        for col_num in range(len(x))])
                y_pred = self.activation_function(y_in)

                if y_pred != y_:
                    self.update_weights(x, y_, learning_rate)
                    y_false += 1
                else:
                    y_true += 1

            accuracy = y_true / (y_true + y_false)
            accuracy_list.append(accuracy)

            if not self.weights_changed(last_weights, last_bias):
                break

        mean_accuracy = sum(accuracy_list) / len(accuracy_list)
        return mean_accuracy

    def activation_function(self, y_in: float):
        """ Activation function used to predict the output of the perceptron."""
        return 1 if y_in >= 0 else -1  # bipolar step function

    def update_weights(self, x: list, y_: int, learning_rate: float):
        """ Updates the weights and bias of the perceptron."""
        self.weights.update(
            {col_num: self.weights[col_num] + (learning_rate * y_ * x[col_num]) for col_num in range(len(x))})
        self.bias += (learning_rate * y_)

    def weights_changed(self, weights: dict, bias: float):
        """ Checks if the weights and bias have changed."""
        return self.weights != weights or self.bias != bias

    def predict(self, X: list):
        """ Predicts the output of the perceptron."""
        predicted = {}

        for x in X:
            y_in = self.bias + \
                sum([x[col_num] * self.weights[col_num]
                    for col_num in range(len(x))])
            y_pred = self.activation_function(y_in)
            
            predicted[f"{str(x)}"] = y_pred

        return predicted
