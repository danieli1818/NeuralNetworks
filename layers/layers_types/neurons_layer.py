from functions.activation_functions.activation_function import ActivationFunction
from functions.activation_functions.sigmoid import Sigmoid
from layers.layer import Layer

import numpy as np


class NeuronsLayer(Layer):

    def __init__(self, output_length: int, activation_function: ActivationFunction
                 , should_sum_gradients=False):
        self.layer_length = None
        self.output_length = output_length
        self.activation_function = activation_function
        self.weights = None
        self.biases = np.array([0 for _ in range(output_length)])
        self.inputs_shape = None
        self.inputs = None
        self.outputs_before_activation_function = None
        self.should_sum_gradients = should_sum_gradients

    def feed_forward(self, inputs):
        self.inputs_shape = inputs.shape
        self.inputs = inputs.flatten()

        if self.layer_length is None:
            self.layer_length = len(self.inputs)
            self.weights = np.array([[1 for _ in range(len(self.inputs))] for _ in range(self.output_length)])

        self.outputs_before_activation_function = self.weights @ self.inputs + self.biases
        return self.activation_function.calculate(self.outputs_before_activation_function)

    def backpropagation(self, outputs_gradients, learning_rate):
        activation_function_derivations = \
            self.activation_function.calculate_derivative(self.outputs_before_activation_function)
        # weights update preprocessing
        if self.should_sum_gradients:
            sum_outputs_gradients = sum(outputs_gradients)
            len_outputs_gradients = len(outputs_gradients)
            outputs_gradients = np.array([sum_outputs_gradients for _ in range(len_outputs_gradients)])
        derivations_matrix = np.repeat(np.expand_dims(activation_function_derivations, axis=1)
                                       , self.layer_length, axis=1)
        inputs_matrix = np.repeat(np.expand_dims(self.inputs, axis=0)
                                       , len(self.weights), axis=0)
        outputs_gradients_matrix = np.repeat(np.expand_dims(outputs_gradients, axis=1)
                                       , self.layer_length, axis=1)
        weights_delta = np.multiply(np.multiply(derivations_matrix, inputs_matrix), outputs_gradients_matrix)

        # biases update preprocessing
        biases_delta = np.multiply(activation_function_derivations, outputs_gradients)

        # inputs gradients preprocessing
        weights_multiply_derivations = np.multiply(self.weights, derivations_matrix)

        # weights updating
        self.weights = np.subtract(self.weights, learning_rate * weights_delta)

        # biases updating
        self.biases = np.subtract(self.biases, learning_rate * biases_delta)

        # returning inputs gradients
        return (weights_multiply_derivations.transpose() @ outputs_gradients).reshape(self.inputs_shape)

if __name__ == "__main__":
    neurons_layer = NeuronsLayer(2, Sigmoid())
    print(neurons_layer.feed_forward(np.array([1, 2])))
    print(neurons_layer.backpropagation(np.array([2, 4]), 0.1))
    print(neurons_layer.feed_forward(np.array([1, 2])))


