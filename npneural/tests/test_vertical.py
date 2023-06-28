import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from activation    import ReLU, TanH
    from layers        import Dense
    from losses        import Loss_MSE
    from wrappers      import Network
    from data.vertical import X,y

except ModuleNotFoundError:
    from npneural.activation    import ReLU, TanH
    from npneural.layers        import Dense
    from npneural.losses        import Loss_MSE
    from npneural.wrappers      import Network
    from npneural.data.vertical import X,y

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler


def run_test():

    sns.set()


    def adjoin(*objs, **kwds):
        # Pretty printing arrays for debugging
        from pandas.io.formats.printing import adjoin as adj
        space = kwds.get('space', 8)
        reprs = [repr(obj).split('\n') for obj in objs]
        print(adj(space, *reprs), '\n\n')


    # Only training labels are one-hot encoded
    OHE = OneHotEncoder(sparse_output=False, dtype='uint8')
    ohce = OHE.fit_transform(np.vstack(y))
    # Maybe overfitting?
    neunet = Network(
        Dense(2,100),
        TanH(),
        Dense(100,3),
        TanH(),
        Dense(3,3),
        TanH(), ReLU(),
    )

    neunet.fit(X, ohce, debug=[10,100,500,1000], epochs=1000)
    print("Final Accuracy: ", end=' ')

    # Predictions

    print(np.mean(np.argmax(neunet.predict(X), axis=1) == y), end='\n\n')
    print()
    print("Predictions, Actual:")
    adjoin(np.argmax(neunet.predict(X), axis=1), y)

    # Visualization

    XX, YY = np.meshgrid(
        np.linspace(X[:,0].min()-1e-1, X[:,0].max()+1e-1, 100),
        np.linspace(X[:,1].min()-1e-1, X[:,1].max()+1e-1, 100)
    )

    fig, axs = plt.subplots(1,2,sharex=True, sharey=True, figsize=(8,4))
    fig.tight_layout()

    ZZ = neunet.predict(np.c_[XX.ravel(), YY.ravel()])
    # ZZ = np.round(np.clip(ZZ, 1e-4, 1-1e-4), 3)
    ZZ = np.argmax(ZZ, axis=1)
    ZZ = ZZ.reshape(XX.shape)

    sns.scatterplot(x=X[:,0], y=X[:,1], c=y, cmap='copper_r', ax=axs[0])
    axs[1].contourf(XX, YY, ZZ, cmap='copper_r');
    plt.show()
