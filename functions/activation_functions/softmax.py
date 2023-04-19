import numpy as np

from functions.activation_functions.activation_function import ActivationFunction
from layers.layers_types.neurons_layer import NeuronsLayer


class Softmax(ActivationFunction):

    def __init__(self):
        pass

    @staticmethod
    def _calculate(num: float, inputs_exponent_sum: float):
        return np.exp(num) / inputs_exponent_sum

    def calculate(self, inputs: list):
        inputs_exponent_sum = sum(np.exp(inputs))
        return np.array([Softmax._calculate(num, inputs_exponent_sum) for num in inputs])

    @staticmethod
    def _calculate_derivative(e_tc_divided_by_s_multiply_by_s, exponents, index, expected_class_index):
        if index != expected_class_index:
            return -e_tc_divided_by_s_multiply_by_s * exponents[index]
        return e_tc_divided_by_s_multiply_by_s * (sum(exponents) - exponents[expected_class_index])

    def calculate_derivative(self, inputs: list):
        expected_class_index = np.argmax(inputs)
        exponents = np.exp(inputs)
        s = sum(exponents)
        e_tc_divided_by_s_multiply_by_s = exponents[expected_class_index] / (s ** 2)
        return np.array([Softmax._calculate_derivative(e_tc_divided_by_s_multiply_by_s
                                           , exponents, i, expected_class_index) for i in range(len(inputs))])

if __name__ == "__main__":
    softmax = Softmax()
    layer = NeuronsLayer(2, softmax, should_sum_gradients=True)
    print(layer.feed_forward(np.array([1, 3])))
    print(layer.backpropagation(np.array([0, 1]), 0.1))
    print(layer.feed_forward(np.array([1, 3])))
