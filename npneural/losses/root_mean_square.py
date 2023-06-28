import numpy as np


# Root Mean Squared Error
# Set as the default Loss function in Network
class Loss_MSE():
    def forward(self, y_pred, y_true):
        # Demands 2D OHE Labels/ Values for performance
        if not (y_pred.shape == y_true.shape and y_true.ndim == 2):
            adjoin(y_pred, y_true)
            raise ValueError("Shape Mismatch between predictions and labels")

        self.diff = y_pred-y_true

        if (self.diff.shape[1] == 1):
            # (N,1)  -> Regression
            self.accu = np.mean(abs(self.diff) < 1e-6)
        else:
            # (N,K)  -> K-class classification, with One-Hot Encoded Labels
            self.accu = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_true, axis=1))

        self.loss = np.mean((self.diff)**2)
        return self.loss

    def backward(self):
        self.grad = 2 * (self.diff) / len(self.diff)
        return self.grad

# For Categorical Cross Entropy,
# Li = -sum(y_true * log(y_pred_prob))
# Li = -log(y_prob[y_true_ohe])
# ∂L/∂X = y_pred_prob - y_true_one_hot
# Gradients are similar to those of MSE, with OHE labels
