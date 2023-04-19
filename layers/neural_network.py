import numpy as np

from functions.activation_functions.sigmoid import Sigmoid
from functions.activation_functions.softmax import Softmax
from functions.cost_functions.cross_entropy import CrossEntropy
from functions.cost_functions.mse import MSE
from layers.costs_layer import CostsLayer
from layers.layers_types.cnn.cnn_filters_layer import CNNFiltersLayer
from layers.layers_types.cnn.pooling_layer import PoolingLayer
from layers.layers_types.neurons_layer import NeuronsLayer

import mnist


class NeuralNetwork:

    def __init__(self, layers, output_layer: CostsLayer):
        self.layers = layers
        self.output_layer = output_layer

    def feed_forward(self, inputs):
        current_inputs = inputs
        for i, layer in enumerate(self.layers):
            current_inputs = layer.feed_forward(current_inputs)
        return current_inputs

    def calc_costs(self, inputs, expected_outputs):
        return self.output_layer.calculate_costs(self.feed_forward(inputs), expected_outputs)

    def backpropagation(self, outputs, expected_outputs, learning_rate=0.1):
        current_outputs_gradients = self.output_layer.calculate_costs_gradients(outputs, expected_outputs)
        for layer in self.layers[-1::-1]:
            current_outputs_gradients = layer.backpropagation(current_outputs_gradients, learning_rate)

    def train_network(self, examples, outputs, learning_rate=0.1, epochs=1):
        for epoch in range(epochs):
            for example, expected_output in zip(examples, outputs):
                output = self.feed_forward(example)
                self.backpropagation(output, expected_output, learning_rate)

# nn = NeuralNetwork([NeuronsLayer(2, Sigmoid()), NeuronsLayer(1, Sigmoid())], CostsLayer(MSE()))
# examples = [np.array([np.random.randint(-100, 100), np.random.randint(-100, 100)]) for _ in range(1000)]
# outputs = [np.array([1]) if x > y else np.array([0]) for x, y in examples]
# nn.train_network(examples, outputs, epochs=10)
# print(nn.feed_forward(np.array([2, 1])))
# print(nn.calc_costs(np.array([2, 1]), np.array([1])))

nn = NeuralNetwork([CNNFiltersLayer(2, (3, 3))
                       , PoolingLayer((2, 2), lambda matrix : matrix.max())
                    , NeuronsLayer(10, Softmax(), should_sum_gradients=True)], CostsLayer(CrossEntropy()))

train_images = mnist.train_images()[:1000]
train_outputs = mnist.train_labels()[:1000]
processed_train_outputs = np.array([])
for train_output in train_outputs:
    processed_train_output = np.full(10, 0)
    processed_train_output[train_output] = 1
    processed_train_outputs = np.append(processed_train_outputs, processed_train_output)
processed_train_outputs = processed_train_outputs.reshape((1000, 10))
nn.train_network(train_images, processed_train_outputs)

