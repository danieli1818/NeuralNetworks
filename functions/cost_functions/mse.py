from functions.cost_functions.cost_function import CostFunction


class MSE(CostFunction):

    def calculate(self, value: float, predicted_value: float):
        return (value - predicted_value) ** 2

    def calculate_derivative(self, value: float, predicted_value: float):
        return -2 * (value - predicted_value)
