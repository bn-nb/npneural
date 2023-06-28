import numpy as np
from sklearn.preprocessing import OneHotEncoder

# https://datascience.stackexchange.com/a/117283
# https://stats.stackexchange.com/q/376312

# Jupyter commands
# mnist_url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"

# ! curl {mnist_url} -o mnist.npz

# Data Preparation
# Flatten (N,28,28) images to (N,784)
# One Hot Encode Training Labels

try:
    data = np.load('data/mnist.npz')
except FileNotFoundError:
    data = np.load('npneural/data/mnist.npz')

X_train = data['x_train'].reshape(-1,784)
X_test  = data['x_test'].reshape(-1,784)

# Only training labels are one-hot encoded
OHE = OneHotEncoder(sparse_output=False, dtype='uint8')
y_train = OHE.fit_transform(np.vstack(data['y_train']))
y_test  = data['y_test']
