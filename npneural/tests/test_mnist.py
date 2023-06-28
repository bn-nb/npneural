import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from activation  import ReLU, TanH
    from layers      import Dense
    from losses      import Loss_MSE
    from wrappers    import Network
    from data.mnist  import X_train, X_test, y_train, y_test
    
except ModuleNotFoundError:
    from npneural.activation  import ReLU, TanH
    from npneural.layers      import Dense
    from npneural.losses      import Loss_MSE
    from npneural.wrappers    import Network
    from npneural.data.mnist  import X_train, X_test, y_train, y_test

def adjoin(*objs, **kwds):
    # Pretty printing arrays for debugging
    from pandas.io.formats.printing import adjoin as adj
    space = kwds.get('space', 8)
    reprs = [repr(obj).split('\n') for obj in objs]
    print(adj(space, *reprs), '\n\n')


def run_test():

    print('\n\n', X_train.shape, X_test.shape, y_train.shape, y_test.shape, '\n\n')
    cmap = sns.color_palette("blend:black,white", as_cmap=True)

    # Print Sample Images

    i0 = np.random.choice(range(len(X_train)))
    i1 = np.random.choice(range(len(X_test)))

    fig, axs = plt.subplots(1, 2, figsize=(8,4), sharey=True)
    fig.tight_layout()

    img0 = X_train[i0].reshape(28,28)
    img1 = X_test[i1].reshape(28,28)
    lab0 = np.argmax(y_train[i0])
    lab1 = y_test[i1]


    axs[0].grid(False)
    axs[0].imshow(img0, cmap=cmap)
    axs[0].set_title(f"Label: {lab0}")

    axs[1].grid(False)
    axs[1].imshow(img1, cmap=cmap)
    axs[1].set_title(f"Label: {lab1}")
    plt.show()

    # Training
    # TanH+ReLU -> attempt to replace Sigmoid+SoftMax

    neunet = Network(
        Dense(784, 20),
        TanH(),
        Dense(20, 10),
        TanH(), ReLU(),
    )

    neunet.fit(X_train, y_train, epochs=1000, debug=[10,50,100,500,1000])

    # Predictions

    pred = np.argmax(neunet.predict(X_test), axis=1)
    print(pred)
    print(f"Test Accuracy: {np.mean(pred == y_test)}")
