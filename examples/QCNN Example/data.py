import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_openml

def load_01(n_components):
    mnist = fetch_openml('mnist_784', version=1)
    mnist.target = mnist.target.astype(int)

    X_filtered = mnist.data[(mnist.target == 0) | (mnist.target == 1)]
    y_filtered = mnist.target[(mnist.target == 0) | (mnist.target == 1)]

    pca = PCA(n_components=n_components)

    X_pca = pca.fit_transform(X_filtered)
    y = np.array(y_filtered)
    return X_pca, y
    


if __name__=="__main__":
    X, y = load_01(256)
    