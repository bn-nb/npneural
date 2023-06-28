import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from activation  import ReLU, TanH
    from layers      import Dense
    from losses      import Loss_MSE
    from wrappers    import Network

except ModuleNotFoundError:
    from npneural.activation  import ReLU, TanH
    from npneural.layers      import Dense
    from npneural.losses      import Loss_MSE
    from npneural.wrappers    import Network

from sklearn.preprocessing import StandardScaler


def run_test():

    sns.set()

    # XOR is the most fundamental nonlinear data
    # Widely used for testing Neural Networks

    X = np.array([[0,0], [0,1], [1,0], [1,1]])
    y = np.array([ 0,1,1,0 ]).reshape(-1,1)

    neunet = Network(
        Dense(2,4, scaler=StandardScaler()),
        Dense(4,4),
        # TanH(),
        ReLU(),
        # Sigmoid(),
        Dense(4,1),
    )

    neunet.fit(X, y, debug=[100,500,1000])

    # Predictions

    print(np.round(np.clip(neunet.predict(X), 1e-4, 1-1e-4), 3))

    # Visualization

    XX, YY = np.meshgrid(
        np.linspace(X[:,0].min()-1e-1, X[:,0].max()+1e-1, 100),
        np.linspace(X[:,1].min()-1e-1, X[:,1].max()+1e-1, 100)
    )

    fig, axs = plt.subplots(1,2,sharex=True, sharey=True, figsize=(8,4))
    fig.tight_layout()

    ZZ = neunet.predict(np.c_[XX.ravel(), YY.ravel()])
    ZZ = np.round(np.clip(ZZ, 1e-4, 1-1e-4), 3)
    # ZZ = np.argmax(ZZ, axis=1)
    ZZ = ZZ.reshape(XX.shape)

    sns.scatterplot(x=X[:,0], y=X[:,1], c=y, cmap='copper_r', ax=axs[0])
    axs[1].contourf(XX, YY, ZZ, cmap='copper_r');
    plt.show()
