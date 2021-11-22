import numpy as np


def activation(x):
    return np.arcsinh(x)


def dactivation(x):
    return 1.0 / np.sqrt(1 + x ** 2)


class network:
    class net_tick:
        def __init__(self, W_IH, W_HH, W_HI):
            self.W_IH = W_IH.copy()
            self.W_HH = W_HH.copy()
            self.W_HI = W_HI.copy()
            self.input = []
            self.context = []
            self.hidden_input_from_context = []
            self.hidden_input_from_input = []
            self.hidden_input_total = []
            self.hidden_output = []
            self.output_from_hidden = []
            self.output_activated = []

        def forward(self, input, context):
            self.input = input.copy()
            self.context = context.copy()
            self.hidden_input_from_context = self.W_HH @ self.context
            self.hidden_input_from_input = self.W_IH @ self.input
            self.hidden_input_total = self.hidden_input_from_input + self.hidden_input_from_context
            self.hidden_output = activation(self.hidden_input_total)
            self.output_from_hidden = self.W_HI @ self.hidden_output
            self.output_activated = self.output_from_hidden
            # self.output_activated = activation(self.output_from_hidden)
            return self.output_activated

    def __init__(self, input_size, hidden_size, output_size, learning_rate=10e-10):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.W_IH = (np.random.rand(self.hidden_size, self.input_size) + 0.01) * 2 - 1
        self.W_HH = (np.random.rand(self.hidden_size, self.hidden_size) + 0.01) * 2 - 1
        self.W_HI = (np.random.rand(self.input_size, self.hidden_size) + 0.01) * 2 - 1

    def create_net(self, size):
        self.unfolded_net = []
        for i in range(size):
            self.unfolded_net.append(self.net_tick(self.W_IH, self.W_HH, self.W_HI))

    def get_context(self, index):
        if index == 0:
            return np.zeros((self.hidden_size, 1))
        else:
            return self.unfolded_net[index - 1].hidden_output

    def train(self, input, output):
        size = len(input)
        self.create_net(size)
        for k in range(1000):
            net_output = self.forward(input)
            error = (output - net_output) ** 2
            self.backwards(size - 1, error)
            print(error)
        for i in range(size):
            print(self.unfolded_net[i].W_IH.sum())


    def forward(self, input):
        ans = []
        for i in range(len(input)):
            ans = self.unfolded_net[i].forward(np.array(input[i], ndmin=2).T, self.get_context(i))
        return ans

    def backwards(self, size, init_error):
        delta_HI = self.learning_rate / 1000000.0 * init_error @ \
                   self.unfolded_net[size].hidden_output.T
        hidden_error = self.unfolded_net[size].W_HI.T @ init_error
        self.unfolded_net[size].W_HI += delta_HI
        for j in range(size, -1, -1):
            delta_HH = self.learning_rate * hidden_error * dactivation(self.unfolded_net[j].hidden_input_total) @ \
                       self.unfolded_net[j - 1].hidden_output.T
            delta_IH = self.learning_rate * hidden_error * dactivation(self.unfolded_net[j].hidden_input_total) @ \
                       self.unfolded_net[j].input.T
            hidden_error = self.unfolded_net[j - 1].W_HH.T @ hidden_error
            self.unfolded_net[j].W_IH += delta_IH
            self.unfolded_net[j].W_HH += delta_HH
