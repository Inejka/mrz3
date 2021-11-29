import numpy as np


def activation(x):
    return np.arcsinh(x)


def dactivation(x):
    return 1.0 / np.sqrt(1 + x ** 2)


class network:

    def __init__(self, input_size, hidden_size, output_size, context_size, learning_rate=10e-10):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.context_size = context_size
        self.learning_rate = learning_rate
        self.W_IH = (np.random.rand(self.hidden_size, self.input_size) + 0.01) * 2 - 1
        self.W_HC = (np.random.rand(self.context_size, self.hidden_size) + 0.01) * 2 - 1
        self.W_HI = (np.random.rand(self.input_size, self.hidden_size) + 0.01) * 2 - 1
        self.W_CH = (np.random.rand(self.context_size, self.hidden_size) + 0.01) * 2 - 1
        self.context = np.zeros((context_size, 1))

    def train(self, sequence):
        for i in range(len(sequence) - 1):
            input = np.array(sequence[i], ndmin=2).T
            output = np.array(sequence[i + 1], ndmin=2).T
            hidden_input_from_input = self.IH @ input
            hidden_input_from_context = self.CH @ self.context
            hidden_total_input = hidden_input_from_input + hidden_input_from_context
            hidden_activated = activation(hidden_total_input)
            next_context = self.HC @ hidden_activated
            net_output = self.HI @ hidden_activated
            error = (net_output - output) ** 2
            d_W_HC = error * self.learning_rate @ net_output.T
            #hidden_error = self.HI @
