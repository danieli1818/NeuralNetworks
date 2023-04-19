from layers.layer import Layer

import numpy as np

from layers.layers_types.cnn.utils.cnn_utils import CNNUtils


class CNNFiltersLayer(Layer):

    def __init__(self, num_of_filters: int, filters_size: tuple):
        filters_size_x = filters_size[0]
        filters_size_y = filters_size[1]
        self.filters = np.array([[[1 for _ in range(filters_size_x)] for _ in range(filters_size_y)]
                                 for _ in range(num_of_filters)])
        self.last_inputs = None

    def feed_forward(self, inputs):
        self.last_inputs = inputs
        return np.array([CNNFiltersLayer.feed_forward_filter(inputs, current_filter)
                         for current_filter in self.filters])

    @staticmethod
    def _calculate_filter_weights_deltas_of_layer_of_inputs(current_filter, current_inputs, output_gradients):
        sub_inputs_matrices_generator = CNNUtils.generate_sub_matrices(current_inputs, current_filter.shape)
        weights_deltas = np.full(current_filter.shape, 0)
        for (row, col), sub_matrix in sub_inputs_matrices_generator:
            weights_deltas = np.add(weights_deltas, output_gradients[row, col] * sub_matrix)
        # print(weights_deltas)
        return weights_deltas

    def _calculate_weights_deltas_filter(self, current_filter, output_gradients):
        weights_deltas = np.full(current_filter.shape, 0)
        if len(self.last_inputs.shape) == 3:
            for inputs_layer, output_gradients_layer in zip(self.last_inputs, output_gradients):
                weights_deltas = np.add(weights_deltas, CNNFiltersLayer.
                                        _calculate_filter_weights_deltas_of_layer_of_inputs(
                    current_filter, inputs_layer, output_gradients_layer))
        else:
            weights_deltas = CNNFiltersLayer.\
                _calculate_filter_weights_deltas_of_layer_of_inputs(current_filter, self.last_inputs, output_gradients)
        return weights_deltas

    @staticmethod
    def _calculate_inputs_deltas_of_layer_filter(inputs_layer, current_filter, output_gradients):
        inputs_deltas = np.full(inputs_layer.shape, 0)
        filter_rows, filter_cols = current_filter.shape
        for (row, col), inputs_sub_matrix in CNNUtils.generate_sub_matrices(inputs_layer, current_filter.shape):
            current_deltas = output_gradients[row, col] * current_filter
            inputs_deltas[row:row + filter_rows, col:col + filter_cols] = \
                np.add(inputs_deltas[row:row + filter_rows, col:col + filter_cols], current_deltas)
        return inputs_deltas

    def _calculate_inputs_deltas_of_layer(self, inputs_layer, output_gradients):
        inputs_deltas = np.full(inputs_layer.shape, 0)
        for current_filter, current_output_gradients in zip(self.filters, output_gradients):
            inputs_deltas = np.add(inputs_deltas
                                   , CNNFiltersLayer._calculate_inputs_deltas_of_layer_filter(
                    inputs_layer, current_filter, current_output_gradients))
        return inputs_deltas

    def _calculate_inputs_deltas(self, output_gradients):
        inputs_deltas = np.full(self.last_inputs.shape, 0)
        last_inputs = self.last_inputs
        if len(last_inputs.shape) == 4:
            last_inputs = last_inputs.reshape((last_inputs[0] * last_inputs[1], last_inputs[2], last_inputs[3]))
        if len(last_inputs.shape) == 3:
            for i, (inputs_layer, output_gradients_layer) in enumerate(zip(last_inputs, output_gradients)):
                np.add(inputs_deltas, self._calculate_inputs_deltas_of_layer(inputs_layer, output_gradients_layer))
        else:
            inputs_deltas = self._calculate_inputs_deltas_of_layer(last_inputs, output_gradients)
        return inputs_deltas

    def backpropagation(self, outputs_gradients, learning_rate):
        new_filters_deltas = np.array([])
        for current_filter, output_gradient_of_filter in zip(self.filters, outputs_gradients):
            current_weights_deltas_filter = self._calculate_weights_deltas_filter(
                current_filter, output_gradient_of_filter)
            new_filters_deltas = np.append(new_filters_deltas, current_weights_deltas_filter)
        new_filters_deltas = new_filters_deltas.reshape(self.filters.shape)
        # print("New filters deltas: ", new_filters_deltas)
        # print("New filters deltas shape: ", new_filters_deltas.shape)
        # print("Filters: ", self.filters)
        # print("Filters shape: ", self.filters.shape)
        inputs_gradients = self._calculate_inputs_deltas(outputs_gradients)
        self.filters = np.subtract(self.filters, learning_rate * new_filters_deltas)
        return inputs_gradients

    @staticmethod
    def feed_forward_filter(inputs, current_filter):
        filter_x_size = len(current_filter[0])
        filter_y_size = len(current_filter)
        results_shape = (inputs.shape[0] - filter_y_size + 1, inputs.shape[1] - filter_x_size + 1)
        results = np.full(results_shape, 0)
        for (row, col), current_inputs in CNNUtils.generate_sub_matrices(inputs, (filter_x_size, filter_y_size)):
            current_matrix = np.multiply(current_inputs, current_filter)
            results[row, col] = sum(current_matrix.flatten())
        return results

cnn_filters_layer = CNNFiltersLayer(2, (3, 3))
print(cnn_filters_layer.feed_forward(np.array([[1] * 4] * 4)))
print(cnn_filters_layer.backpropagation(np.array([[[2] * 2] * 2] * 2), 0.1))
