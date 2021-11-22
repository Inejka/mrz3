import numpy as np


class network:
    def __init__(self, input_neuron_size, hidden_neuron_size, output_neuron_size, context_neuron_size,
                 learning_rate=10e-6):
        self.input_neuron_size = input_neuron_size
        self.hidden_neuron_size = hidden_neuron_size
        self.output_neuron_size = output_neuron_size
        self.context_neuron_size = context_neuron_size
        self.learning_rate = learning_rate
        self.context_layer = np.zeros(context_neuron_size)
        self.unfolded_net = []
        self.init_weight()

    def init_weight(self):
        self.W_IH = (np.random.rand(self.hidden_neuron_size, self.input_neuron_size) + 0.01) * 2 - 1
        self.W_HO = (np.random.rand(self.output_neuron_size, self.hidden_neuron_size) + 0.01) * 2 - 1
        self.W_CH = (np.random.rand(self.hidden_neuron_size, self.context_neuron_size) + 0.01) * 2 - 1
        self.W_HC = (np.random.rand(self.context_neuron_size, self.hidden_neuron_size) + 0.01) * 2 - 1

    def train(self, input, output):
        self.unfold(len(input))
        self.unfolded_net[0].context = np.zeros((self.context_neuron_size, 1))
        for k in range(20):
            for i in range(len(input)):
                for j in range(i + 1):
                    self.unfolded_net[j].input = np.array(input[j], ndmin=2).T
                    self.unfolded_net[j].forward()
                    self.unfolded_net[j + 1].context = self.unfolded_net[j].context_output_from_hidden
                init_error = np.power(np.array(output[i], ndmin=2).T - self.unfolded_net[i].output_output_from_hidden,
                                      2)
                error = init_error
                for j in range(i, -1, -1):
                    error = self.unfolded_net[j].back_prop(error, self.learning_rate, j == i,
                                                           self.unfolded_net[j - 1].hidden_input_total)
                print(i, init_error)

    def unfold(self, size):
        for i in range(-1, size + 1):
            self.unfolded_net.append(self.net_in_tick(self.W_IH, self.W_HO, self.W_CH, self.W_HC))

    class net_in_tick:
        def __init__(self, W_IH, W_HO, W_CH, W_HC):
            self.context = []
            self.input = []
            self.hidden_input_from_context = []
            self.hidden_input_from_input = []
            self.hidden_input_total = None
            self.hidden_output = []
            self.output_output_from_hidden = []
            self.context_output_from_hidden = []
            self.W_IH = W_IH.copy()
            self.W_HO = W_HO.copy()
            self.W_CH = W_CH.copy()
            self.W_HC = W_HC.copy()

        def forward(self):
            self.hidden_input_from_context = self.W_CH @ self.context
            self.hidden_input_from_input = self.W_IH @ self.input
            self.hidden_input_total = self.hidden_input_from_input + self.hidden_input_from_context
            self.hidden_output = np.arcsinh(self.hidden_input_total)
            self.context_output_from_hidden = self.W_HC @ self.hidden_input_total
            self.output_output_from_hidden = self.W_HO @ self.hidden_output

        def back_prop(self, error, learning_rate, last_layer, f_state=None):
            if last_layer:
                self.delta_W_HO = learning_rate * error @ self.hidden_output.T
                self.error_hidden_layer = self.W_HO.T @ error
            else:
                self.error_hidden_layer = self.W_HC.T @ error
            self.delta_W_IH = learning_rate * self.error_hidden_layer * (1.0 / np.sqrt(
                1 + self.hidden_input_total * self.hidden_input_total)) @ self.input.T
            self.delta_W_CH = learning_rate * self.error_hidden_layer * (1.0 / np.sqrt(
                1 + self.hidden_input_total * self.hidden_input_total)) @ self.context.T
            self.context_error = self.W_CH.T @ self.error_hidden_layer
            if f_state is not None:
                self.delta_W_HC = learning_rate * self.context_error @ f_state.T
                self.W_HC += self.delta_W_HC
            if last_layer:
                self.W_HO += self.delta_W_HO
            self.W_CH += self.delta_W_CH
            self.W_IH += self.delta_W_IH
            return self.context_error
