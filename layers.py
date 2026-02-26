import numpy as np

class Linear:
    def __init__(self, input_dim, output_dim):
        # Kaiming initialization, better for (Leaky) ReLU. W ~ N(0, 2/n_in)
        self.W = np.random.randn(input_dim, output_dim) * np.sqrt(2.0 / input_dim)
        self.b = np.zeros(output_dim)
        self.params = [self.W, self.b]
        self.grads = [np.zeros_like(self.W), np.zeros_like(self.b)]  # create gradient (dL/dw, dL/db) placeholders initially filled with zeros, with same dims as W and b
        self.input = None
    
    def forward(self, x):
        self.input = x
        return np.dot(x, self.W) + self.b # z = Wx + b

    def backward(self, grad_output):
        self.grads[0] = np.dot(self.input.T, grad_output)   # dL/dW = a(l-1) x 1 x dL/da(l)
        self.grads[1] = np.sum(grad_output, axis=0)         # dL/db = dL/da(l)
        return np.dot(grad_output, self.W.T)                # return dL/da(l-1) for backprop

class ReLU:
    def __init__(self):
        self.params = []
        self.grads = []
        self.input = None
    
    def forward(self, x):
        self.input = x
        return np.maximum(0, x)

    def backward(self, grad_output):
        relu_grad = (self.input>0).astype(float)        # grad = 1 if x>0, 0 else
        return grad_output * relu_grad

class SoftmaxCrossEntropy:
    def __init__(self):
        self.params = []
        self.grads = []
        self.probs = None       # probabilities
        self.y = None
    
    def forward(self, logits, y):
        # logits: (batch size, num_classes)
        # y: (batch_size), integer label
        self.y = y
        shift_logits  = logits - np.max(logits, axis=1, keepdims=True)          # numerical stability (subtract max value from all inputs without changing the output probs)
        exp_logits = np.exp(shift_logits)                                       # e^(Xi)
        self.probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)     # Softmax formula (sum on cols)
        
        batch_size = logits.shape[0]
        
        # cross entropy: CE = - log(P_y)
        log_probs = - np.log(self.probs[np.arange(batch_size), y] + 1e-9)      # add small epsilon to avoid log(0)
        loss = np.sum(log_probs) / batch_size
        return loss
    
    def backward(self):
        batch_size = self.probs.shape[0]
        grad = self.probs.copy()
        # gradient of loss with respect to logits z --> dL/dz = delta(L) = a(L) - y = Pi - yi
        grad[np.arange(batch_size), self.y] -= 1       # subtract 1 only at the correct class index
        return grad / batch_size
