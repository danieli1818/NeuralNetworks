# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from functions.activation_functions.sigmoid import Sigmoid
from functions.cost_functions.mse import MSE
from neural_network import NeuralNetwork
from neuron import Neuron

import random as rnd


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

def run_neural_network():
    # Define dataset
    data = [
        [-2, -1],  # Alice
        [25, 6],  # Bob
        [17, 4],  # Charlie
        [-15, -6],  # Diana
    ]
    all_y_trues = [
        [1],  # Alice
        [0],  # Bob
        [0],  # Charlie
        [1],  # Diana
    ]
    # print_hi('PyCharm')
    neural_network = NeuralNetwork([2, 2, 1], [Sigmoid() for _ in range(3)], [MSE()])
    # training_examples = [[rnd.randint(-100, 100), rnd.randint(-100, 100)] for _ in range(1000)]
    # expected_results = [[1] if example[0] > example[1] else [0] for example in training_examples]
    # for _ in range(991):
    #     neural_network.train_network(data, all_y_trues)
    # neural_network.feed_forward([-2, -1])
    # output = neural_network.get_outputs()[0]
    # print(f"Output: {output}, Output Delta: {output}")
    epochs = 1000
    num_of_examples = 100
    examples = [[rnd.randint(-100, 100), rnd.randint(-100, 100)] for _ in range(num_of_examples)]
    for i in range(epochs):
        results = [[1] if x > y else [0] for x, y in examples]
        neural_network.train_network(examples, results, learning_rate=0.1)
        average_deltas = 0
        for example, result in zip(examples, results):
            neural_network.feed_forward(example)
            output = neural_network.get_outputs()[0]
            average_deltas += abs(result[0] - output)
        average_deltas /= num_of_examples
        print(f"Epoch {i} Average Deltas: {average_deltas * 100}%")

    # neuron = Neuron(activation_function=Sigmoid())
    # neuron.update_bias(4)
    # print(neuron.feed_forward([0, 1], [2, 3]))

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    run_neural_network()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/



