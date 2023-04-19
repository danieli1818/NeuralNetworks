import numpy as np

from layers.layer import Layer
from layers.layers_types.cnn.utils.cnn_utils import CNNUtils


class PoolingLayer(Layer):

    def __init__(self, pooling_size, pooling_function):
        self.pooling_size = pooling_size
        self.pooling_function = pooling_function
        self.last_inputs = None

    def feed_forward(self, inputs):
        self.last_inputs = inputs
        if len(inputs.shape) == 4:
            inputs = inputs.reshape(inputs.shape[0] * inputs.shape[1], inputs.shape[2], inputs.shape[3])
        if len(inputs.shape) == 3:
            result = np.array([])
            for inputs_layer in inputs:
                layer_results = np.array([self.pooling_function(matrix) for _, matrix in
                           CNNUtils.generate_non_colliding_matrices(inputs_layer, self.pooling_size)])
                inputs_rows, inputs_columns = inputs_layer.shape
                layer_results = layer_results.reshape(
                    inputs_rows // self.pooling_size[0], inputs_columns // self.pooling_size[1])
                result = np.append(result, layer_results)
            num_of_filters, inputs_rows, inputs_columns = inputs.shape
            result = result.reshape((num_of_filters
                                     , inputs_rows // self.pooling_size[0], inputs_columns // self.pooling_size[1]))
        else:
            result = np.array([self.pooling_function(matrix) for _, matrix in
                               CNNUtils.generate_non_colliding_matrices(inputs, self.pooling_size)])
            inputs_rows, inputs_columns = inputs.shape
            result = result.reshape((inputs_rows // self.pooling_size[0], inputs_columns // self.pooling_size[1]))
        return result

    def backpropagation(self, outputs_gradients, learning_rate):
        inputs = self.last_inputs
        if len(inputs.shape) == 4:
            inputs = inputs.reshape(inputs.shape[0] * inputs.shape[1], inputs.shape[2], inputs.shape[3])
        if len(inputs.shape) == 3:
            result = np.array([])
            for inputs_layer, outputs_gradients_layer in zip(inputs, outputs_gradients):
                current_result = np.full(inputs_layer.shape, 0)
                num_of_rows_per_sub_matrix = self.pooling_size[1]
                num_of_cols_per_sub_matrix = self.pooling_size[0]
                for indexes, matrix in CNNUtils.generate_non_colliding_matrices(inputs_layer, self.pooling_size):
                    row, col = indexes
                    results_row, results_col = row // num_of_rows_per_sub_matrix, col // num_of_cols_per_sub_matrix
                    max_index = matrix.argmax()
                    row_offset = max_index // num_of_rows_per_sub_matrix
                    col_offset = max_index % num_of_rows_per_sub_matrix
                    current_result[row + row_offset, col + col_offset] = outputs_gradients_layer[results_row, results_col]
                result = np.append(result, current_result)
        else:
            result = np.full(inputs.shape, 0)
            num_of_rows_per_sub_matrix = self.pooling_size[1]
            num_of_cols_per_sub_matrix = self.pooling_size[0]
            for indexes, matrix in CNNUtils.generate_non_colliding_matrices(inputs, self.pooling_size):
                row, col = indexes
                results_row, results_col = row // num_of_rows_per_sub_matrix, col // num_of_cols_per_sub_matrix
                max_index = matrix.argmax()
                row_offset = max_index // num_of_rows_per_sub_matrix
                col_offset = max_index % num_of_rows_per_sub_matrix
                result[row + row_offset, col + col_offset] = outputs_gradients[results_row, results_col]
        result = result.reshape(self.last_inputs.shape)
        return result


pooling_layer = PoolingLayer((2, 2), lambda matrix : matrix.max())
print(pooling_layer.feed_forward(np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])))
print(pooling_layer.backpropagation(np.array([[1] * 2] * 2), 0.1))
