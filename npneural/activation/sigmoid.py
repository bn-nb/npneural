import numpy as np
from .base import Activation


class Sigmoid(Activation):
    def __init__(self):
        super().__init__(self.sigmoid, self.d_dx)

    def sigmoid(self, inputs):
        try:
            self.exps = np.exp(-inputs)
        except FloatingPointError:
            print(inputs)
        return 1/(1+self.exps)
