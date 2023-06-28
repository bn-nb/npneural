import numpy as np
from .base import Activation


class SoftMax(Activation):
    def __init__(self):
        super().__init__(self.softmax, lambda x: x)

    def softmax(self, inputs):
        # Handle Overflow
        inputs -= np.max(input, axis=1, keepdims=True)
        # Handle Underflow
        inputs  = np.clip(input, a_min=-7, a_max=0)
        exps = np.exp(inputs)

        self.soft_op = exps/np.sum(exps, axis=1, keepdims=True)
        return self.soft_op


# Generalization of sigmoid to vectors
# SoftMax -> Vector Output -> "Gradient" is meaningless
# Derivative -> Jacobian. We will simply pass
# output gradients as inputs gradients
