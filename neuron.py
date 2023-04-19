from functions.activation_functions.activation_function import ActivationFunction

import numpy as np

class Neuron:

    def __init__(self, activation_function: ActivationFunction, next_layer_size: int=0):
        self.activation_function = activation_function
        self.after_data = None
        self.before_data = None
        self.back_prop_data = None
        self.weights = [1 for _ in range(next_layer_size)]
        self.bias = 1

    def update_weight(self, i: int, weight):
        if i < 0 or i >= len(self.weights):
            return False
        self.weights[i] = weight
        return True

    def update_weights(self, weights):
        self.weights = weights

    def update_bias(self, bias):
        self.bias = bias

    def get_weight(self, i: int):
        if i < 0 or i >= len(self.weights):
            return None
        return self.weights[i]

    def get_weights(self):
        return self.weights

    def get_bias(self):
        return self.bias

    def set_bias(self, bias):
        self.bias = bias
        return self.bias

    def get_after_data(self):
        return self.after_data

    def set_after_data(self, data: float):
        self.after_data = data

    def get_before_data(self):
        return self.before_data

    def set_before_data(self, data: float):
        self.before_data = data

    def get_back_prop_data(self):
        return self.back_prop_data

    def set_back_prop_data(self, data: float):
        self.back_prop_data = data

    def feed_forward(self, weights: list, inputs: list):
        self.before_data = np.dot(weights, inputs) + self.bias
        self.after_data = self.activation_function.calculate(self.before_data)
        return self.after_data

    def calc_derivation(self, num: float):
        return self.activation_function.calculate_derivative(num)

    def __str__(self):
        return f"Activation Function: {self.activation_function}, Data: {self.after_data}" \
               f", Weights: {self.weights}, Bias: {self.bias}"




