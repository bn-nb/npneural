import numpy as np

try:
    from losses  import Loss_MSE
except ModuleNotFoundError:
    from npneural.losses import Loss_MSE


class Network:
    """
    1. Propagate forward through all layers
    2. Calculate Cost, Gradient of Cost
    3. Regularize Gradients
    4. Backpropagate through all layers
    """

    def __init__(self, *layers):
        self.layers = layers
        self.loss_f = Loss_MSE()

    def predict(self, X):
        assert X.ndim==2, "X has to be a 2D array"
        self.out = X
        for L in self.layers:
            self.out = L.forward(self.out)
        return self.out

    def backprop(self, alpha):
        # Expects self.grad to be defined in fit
        # We are using same learning-rate for all layers
        for L in self.layers[::-1]:
            self.grad = L.backward(self.grad, alpha)

    def gradAutoClip(self, iteration, clip_p):
        # Ref: github.com/pseeth/autoclip
        norm = np.sum( self.grad ** 2 ) ** 0.5
        self.grad_norms[iteration] = norm
        norm_p = np.percentile(self.grad_norms, clip_p)

        if (norm > norm_p and norm_p != 0):
            self.grad *= norm_p/norm

    def score(self, X=None, y=None, loss_z=Loss_MSE(), *args, **kwargs):
        "Returns Accuracy"
        if ((X is None) or (y is None)):
            return self.loss_f.accu

        # Create disposable loss function class
        # to prevent changing state of self.loss_f
        pred = self.predict(X)
        loss_z.forward(pred, y)
        return loss_z.accu


    def fit(self, X, y, *, alpha=5e-2, epochs=1000, clip_p=20, debug=[]):
        # For AutoClipping gradients
        assert X.ndim==2, "X has to be a 2D array"
        self.grad_norms = np.zeros(shape=(epochs,))

        for i in range(epochs):
            pred = self.predict(X)
            self.loss = self.loss_f.forward(pred, y)
            self.grad = self.loss_f.backward()
            self.gradAutoClip(i, clip_p)
            self.backprop(alpha)

            if i+1 in debug:
                with np.printoptions(precision=4):
                    # https://stackoverflow.com/a/59158545/
                    # https://stackoverflow.com/a/71801560/
                    trunc = int(np.ceil(np.log10(epochs)) + 1)
                    print(f"Epoch: {i+1:{trunc}d}",end='    ')
                    print(f"Loss : {self.loss :.5e}",end='    ')
                    print(f"Accuracy : {self.score():.5e}")
                    print('-'*(55+trunc))
