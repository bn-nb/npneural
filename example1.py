import numpy as np

from npneural.wrappers   import Network
from npneural.layers     import Dense
from npneural.activation import ReLU, TanH

import npneural.tests as T

# Clear __pycache__
# https://stackoverflow.com/a/41386937

if __name__ == "__main__":
    # 1 minute
    # T.test_xor()

    # 5 minutes
    # T.test_vertical()

    # 5 minutes
    T.test_spiral()

    # 1 hour
    # T.test_mnist()

