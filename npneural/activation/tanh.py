import numpy as np
from .base import Activation


class TanH(Activation):
    def __init__(self):
        d_dx = lambda x: 1 - np.tanh(x) ** 2
        super().__init__(np.tanh, d_dx)
