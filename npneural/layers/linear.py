import numpy as np

class Dense():
    def __init__(self, n_input_features, n_neurons, scaler=None):
        # Random weights -> all outputs nearly equal
        self.i = n_input_features
        self.j = n_neurons

        # scaler must implement fit_transform
        self.scaler = scaler

        # Glorot Initialization
        params_dict = {
            'low' : -np.sqrt(10/self.i),
            'high':  np.sqrt(10/self.i),
            'size': (self.i, self.j),
        }

        self.weights = np.random.uniform(**params_dict)
        self.biases  = np.zeros(shape=(1, self.j))

    def __str__(self):
        return f"Dense({self.i} -> {self.j})"

    def forward(self, inputs):
        self.inputs  = inputs
        self.out = inputs @ self.weights + self.biases

        if (self.scaler != None):
            self.out = self.scaler.fit_transform(self.out)
        return self.out

    def backward(self, output_gradients, alpha):
        """
        1. ∂C/∂X = ∂C/∂y @ W.T  -> B4 changing W & B
        2. ∂C/∂W = X.T @ ∂C/∂y
        3. ∂C/∂B = ∂C/∂y
        """
        self.outG = output_gradients
        self.inpG = self.outG @ self.weights.T
        self.weights -= alpha * (self.inputs.T @ self.outG)
        self.biases  -= np.sum(alpha * self.outG, axis=0).reshape(1,-1)
        return self.inpG
