import numpy as np


class network:
    def __init__(self, input_neuron_size, hidden_neuron_size, output_neuron_size, context_neuron_size):
        self.input_neuron_size = input_neuron_size
        self.hidden_neuron_size = hidden_neuron_size
        self.output_neuron_size = output_neuron_size
        self.context_neuron_size = context_neuron_size
        self.context_layer = np.zeros(context_neuron_size)
        self.init_weight()

    def init_weight(self):
        self.W_IH = (np.random.rand(self.input_neuron_size, self.hidden_neuron_size) + 0.01) * 2 - 1
        self.W_HO = (np.random.rand(self.hidden_neuron_size, self.output_neuron_size) + 0.01) * 2 - 1
        self.W_CH = (np.random.rand(self.context_neuron_size, self.hidden_neuron_size) + 0.01) * 2 - 1
        self.W_HC = (np.random.rand(self.hidden_neuron_size, self.context_neuron_size) + 0.01) * 2 - 1

    def train(self, input, output):
        hidden_values = input @ self.W_IH
        hidden_values = np.arcsinh(hidden_values)
