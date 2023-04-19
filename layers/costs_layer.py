import numpy as np

from functions.cost_functions.cost_function import CostFunction


class CostsLayer:

    def __init__(self, cost_function: CostFunction):
        self.cost_function = cost_function

    def calculate_costs(self, outputs, expected_outputs):
        cost_function_calc = np.vectorize(self.cost_function.calculate)
        return cost_function_calc(outputs, expected_outputs)

    def calculate_costs_gradients(self, outputs, expected_outputs):
        # cost_function_gradients_calc = np.vectorize(self.cost_function.calculate_derivative)
        # return cost_function_gradients_calc(outputs, expected_outputs)
        return [self.cost_function.calculate_derivative(expected_output, output)
                for output, expected_output in zip(outputs, expected_outputs)]
