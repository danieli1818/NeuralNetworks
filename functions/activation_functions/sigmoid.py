import math

import numpy as np

from functions.activation_functions.activation_function import ActivationFunction

class Sigmoid(ActivationFunction):

    def __init__(self):
        pass

    @staticmethod
    def _calculate(num: float):
        if num > 700:
            return 1
        if num < -300:
            return 0
        data = 1 / (1 + math.e ** (-num))
        return data

    def calculate(self, inputs: list):
        return np.vectorize(Sigmoid._calculate)(inputs)

    @staticmethod
    def _calculate_derivative(num: float):
        calc_result = Sigmoid._calculate(num)
        return calc_result * (1 - calc_result)

    def calculate_derivative(self, inputs: list):
        return np.vectorize(Sigmoid._calculate_derivative)(inputs)
