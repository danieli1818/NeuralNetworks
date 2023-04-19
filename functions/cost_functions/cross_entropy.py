import numpy as np

from functions.cost_functions.cost_function import CostFunction


class CrossEntropy(CostFunction):

    def __init__(self):
        pass

    def calculate(self, value: float, predicted_value: float):
        if value == 0:
            return 0
        return -np.log(predicted_value)

    def calculate_derivative(self, value: float, predicted_value: float):
        if value == 0:
            return 0
        return -1 / predicted_value
