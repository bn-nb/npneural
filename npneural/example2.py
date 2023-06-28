from wrappers   import Network
from layers     import Dense
from activation import ReLU, TanH

if __name__ == "__main__":

    # For testing inside the package
    
    import tests as T

    # Training time is indicated above each test

    # To reduce training time, reduce epochs parameter 
    # in neunet.fit(...) in corresponding test's 
    # source file in npneural/tests/

    # Clear __pycache__
    # https://stackoverflow.com/a/41386937

    # 1 minute
    # T.test_xor()

    # 5 minutes
    # T.test_vertical()

    # 5 minutes
    T.test_spiral()

    # 1 hour
    # T.test_mnist()

