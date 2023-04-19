from neuron import Neuron

import numpy as np


class NeuralNetwork:

    def __init__(self, layers_sizes: list, activation_functions_of_layers: list, cost_functions: list):
        if len(layers_sizes) != len(activation_functions_of_layers):
            raise Exception("Invalid arguments!"
                            "layers_sizes and activation_functions_of_layers must have the same length")
        if layers_sizes[-1] != len(cost_functions):
            raise Exception("Invalid arguments!"
                            "The last layer size must be equal to the cost_functions length")
        self.layers = []
        num_of_layers = len(layers_sizes)
        for layer_index, layer_size in enumerate(layers_sizes):
            if layer_index + 1 == num_of_layers:
                next_layer_size = 0
            else:
                next_layer_size = layers_sizes[layer_index + 1]
            self.layers.append([Neuron(activation_function=activation_functions_of_layers[layer_index]
                                       , next_layer_size=next_layer_size)
                                for _ in range(layer_size)])
        self.cost_functions = cost_functions

    def feed_forward(self, inputs):
        for neuron, neuron_data in zip(self.layers[0], inputs):
            neuron.set_after_data(data=neuron_data)
        for current_layer_index in range(1, len(self.layers)):
            current_inputs = self._get_after_datas_of_layer(current_layer_index - 1)
            layer = self.layers[current_layer_index]
            for neuron_index, neuron in enumerate(layer):
                neuron.feed_forward(self._get_weights_to_neuron(current_layer_index, neuron_index), current_inputs)

    def get_outputs(self):
        return [output_neuron.get_after_data() for output_neuron in self.layers[-1]]

    def back_propagation(self, outputs):
        for neuron, cost_function, output in zip(self.layers[-1], self.cost_functions, outputs):
            neuron.set_back_prop_data(data=cost_function.calculate_derivative(value=output
                                                                           , predicted_value=neuron.get_after_data()))
        for current_layer_index in range(len(self.layers) - 2, -1, -1):
            layer = self.layers[current_layer_index]
            for neuron_index, neuron in enumerate(layer):
                # derivation_till_now = float(np.dot(neuron.get_weights()
                #                          , current_inputs))
                next_layer = self.layers[current_layer_index + 1]
                derivation_value = 0
                for next_neuron_index, next_neuron in enumerate(next_layer):
                    derivation_till_now = next_neuron.get_back_prop_data()
                    current_derivation = neuron.get_weight(next_neuron_index) \
                                         * next_neuron.calc_derivation(next_neuron.get_before_data())
                    derivation_value += derivation_till_now * current_derivation
                neuron.set_back_prop_data(derivation_value)
        # print(f"o1 derivation: {self.layers[2][0].get_back_prop_data()}")

    def train_network(self, inputs_datas: list, outputs_datas: list, learning_rate: float=0.1):
        for input_datas, output_datas in zip(inputs_datas, outputs_datas):
            self._train_network(input_datas, output_datas, learning_rate)

    def _train_network(self, inputs, outputs, learning_rate):
        self.feed_forward(inputs)
        self.back_propagation(outputs)
        for layer_index in range(len(self.layers) - 1, -1, -1):
            self._update_weights_before_layer(layer_index, learning_rate)
            if layer_index != 0:
                self._update_biases_of_layer(layer_index, learning_rate)

    def _update_weights_before_layer(self, layer_index: int, learning_rate):
        layer_before = self.layers[layer_index - 1]
        layer_after = self.layers[layer_index]
        for neuron_before in layer_before:
            for weight_index, weight in enumerate(neuron_before.get_weights()):
                neuron_after = layer_after[weight_index]
                derivation_till_now = neuron_after.get_back_prop_data()
                current_derivation = neuron_before.get_after_data() \
                                     * neuron_after.calc_derivation(neuron_after.get_before_data())
                delta = derivation_till_now * current_derivation
                # if layer_index == 2:
                #     # print(f"w5/w6: {current_derivation}")
                # if layer_index == 1:
                #     if weight_index == 1:
                #         # print(f"w1/w2: {current_derivation}")
                #     else:
                #         # print(f"w3/w4: {current_derivation}")
                updated_weight = weight - learning_rate * delta
                # if layer_index == 2:
                #     # print(f"Updated w5/w6: {updated_weight}")
                # if layer_index == 1:
                #     if weight_index == 1:
                #         # print(f"Update w1/w2: {updated_weight}")
                #     else:
                #         # print(f"Update w3/w4: {updated_weight}")
                neuron_before.update_weight(weight_index, updated_weight)

    def _update_biases_of_layer(self, layer_index: int, learning_rate):
        layer = self.layers[layer_index]
        for neuron in layer:
            derivation_till_now = neuron.get_back_prop_data()
            current_derivation = neuron.calc_derivation(neuron.get_before_data())
            # current_derivation = neuron.calc_derivation(neuron.get_after_data())
            # if layer_index == 2:
            #     # print(neuron.get_before_data())
            #     # print(f"Bias o1: {current_derivation}")
            # if layer_index == 1:
            #     # print(neuron.get_before_data())
            #     # print(f"Bias h1/h2: {current_derivation}")
            delta = derivation_till_now * current_derivation
            updated_bias = neuron.get_bias() - learning_rate * delta
            neuron.set_bias(updated_bias)
            # if layer_index == 2:
            #     # print(f"Bias updated o1: {updated_bias}")
            # if layer_index == 1:
            #     # print(f"Bias updated h1/h2: {updated_bias}")

    def _get_after_datas_of_layer(self, layer_index: int):
        if layer_index < 0 or layer_index >= len(self.layers):
            return None
        return [neuron.get_after_data() for neuron in self.layers[layer_index]]

    def _get_back_prop_datas_of_layer(self, layer_index: int):
        if layer_index < 0 or layer_index >= len(self.layers):
            return None
        return [neuron.get_back_prop_data() for neuron in self.layers[layer_index]]

    def _get_weights_to_neuron(self, layer_index: int, neuron_index: int):
        if layer_index < 0 or layer_index >= len(self.layers):
            return None
        prev_layer = self.layers[layer_index - 1]
        return [neuron.get_weight(neuron_index) for neuron in prev_layer]

    def _get_weights_from_neuron(self, layer_index: int, neuron_index: int):
        if layer_index <= 0 or layer_index > len(self.layers):
            return None
        current_layer = self.layers[layer_index]
        if len(current_layer) <= neuron_index:
            return None
        current_neuron = current_layer[neuron_index]
        return current_neuron.get_weights()



