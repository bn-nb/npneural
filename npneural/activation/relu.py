import numpy as np
from .base import Activation


class ReLU(Activation):
    def __init__(self):
        # np.max is diff from np.maximum
        relu = lambda x: np.maximum(0, x)
        d_dx = lambda x: (x > 0)
        super().__init__(relu, d_dx)
