

class CNNUtils:

    @staticmethod
    def generate_sub_matrices(inputs, sub_matrix_size):
        inputs_rows, inputs_cols = inputs.shape
        sub_matrix_rows, sub_matrix_cols = sub_matrix_size
        for row in range(inputs_rows - sub_matrix_rows + 1):
            for col in range(inputs_cols - sub_matrix_cols + 1):
                yield (row, col), inputs[row:row + sub_matrix_rows, col:col + sub_matrix_cols]

    @staticmethod
    def generate_non_colliding_matrices(inputs, sub_matrix_size):
        inputs_rows, inputs_cols = inputs.shape
        sub_matrix_rows, sub_matrix_cols = sub_matrix_size
        for row in range(0, inputs_rows, sub_matrix_rows):
            for col in range(0, inputs_cols, sub_matrix_cols):
                yield (row, col), inputs[row:row + sub_matrix_rows, col:col + sub_matrix_cols]
