class Activation():
    def __init__(self, func, d_dx):
        self.biases = [[0]]  # Compatibility
        self.activation = func
        self.derivative = d_dx

    def forward(self, inputs):
        self.inputs = inputs
        return self.activation(inputs)

    def backward(self, output_gradient, alpha):
        """
        ∂C/∂X = ∂C/∂y ⊙ F'(X)
        """
        return output_gradient * self.derivative(self.inputs)
