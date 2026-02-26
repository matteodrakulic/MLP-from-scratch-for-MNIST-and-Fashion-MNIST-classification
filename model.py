# manages the list of layers and handles the forward - backward passes

import numpy as np
from layers import Linear, ReLU, SoftmaxCrossEntropy

class MLP:
    def __init__(self, input_dim, hidden_dims, output_dim):
        self.layers = []

        # Create hidden layers
        prev_dim = input_dim
        for h_dim in hidden_dims:
            self.layers.append(Linear(prev_dim, h_dim))
            self.layers.append(ReLU())
            prev_dim = h_dim
        
        # Create output layer
        self.layers.append(Linear(prev_dim, output_dim))
        self.criterion = SoftmaxCrossEntropy()

    def forward(self, x, y):
        # x = input data (batch_size, input_dim)
        # y = labels (batch_size)
        # must return loss and accuracy

        # forward pass through all layers
        out = x
        for layer in self.layers:
            out = layer.forward(out)

        loss = self.criterion.forward(out, y)       # calculate loss

        preds = np.argmax(self.criterion.probs, axis=1)     # find the prediction given by model
        accuracy = np.mean(preds == y)

        return loss, accuracy
    
    def backward(self):
        # backprop through all layers
        grad = self.criterion.backward()
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def step(self, learning_rate):
        for layer in self.layers:
            if hasattr(layer, 'params'):
                for i in range(len(layer.params)):
                    layer.params[i] -= learning_rate * layer.grads[i]
