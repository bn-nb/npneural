import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class SampleClassData:

    def __init__(self, samples_per_class=100, no_of_classes=3, *, dist=None, plot_axis=None):
        # np.random.seed(0)
        self.N = samples_per_class
        self.M = no_of_classes
        self.X = np.zeros(shape=(self.N*self.M, 2))
        self.y = np.zeros(shape=(self.N*self.M,), dtype='uint8')

        funcs = {'S': self.spiral, 'V': self.vertical}

        assert dist in funcs, "Specify data class"
        funcs.get(dist, 'V')()

        if (plot_axis != None):
            sns.scatterplot(x=self.X[:,0], y=self.X[:,1], c=self.y, cmap='Dark2', ax=plot_axis)

    def spiral(self):
        # 2D spiral -> angular points from 'growing' circles

        for i in range(self.M):
            index = range(self.N*i, self.N*(i+1))
            radii = np.linspace(0, 1, self.N)  # Increasing Radii
            arcsz = (np.pi * 2) / self.M       # For symmetrical alignment
            theta = np.linspace(i * arcsz, (i+1) * arcsz, self.N)
            noise = np.random.randn(self.N) * 0.2
            angle = theta + noise

            self.X[index] = np.vstack((radii * np.sin(angle), radii * np.cos(angle))).T
            self.y[index] = i

    def vertical(self):
        for i in range(self.M):
            index = range(self.N*i, self.N*(i+1))
            _x = np.random.randn(self.N) * 0.05 + i/4
            _y = np.random.randn(self.N) * 0.1  + 0.5

            self.X[index] = np.vstack((_x, _y)).T
            self.y[index] = i

if __name__ == "__main__":
    sns.set()
    fig, axs = plt.subplots(1,2, sharex=False, sharey=False, figsize=(8,4))
    fig.tight_layout(pad=3)
    a=SampleClassData(dist='V', plot_axis=axs[0]);
    b=SampleClassData(dist='S', plot_axis=axs[1]);
    plt.show()
