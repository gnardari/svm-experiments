import numpy as np
from sklearn.decomposition import PCA

def pca(dataset, explained_var):
    dataset -= np.mean(dataset, axis=0, dtype=np.uint8)
    # eigen
    cov_matrix = np.cov(dataset.T)
    eigenvals, eigenvects = np.linalg.eigh(cov_matrix)
    # selecionando atributos
    eigenvals = eigenvals/np.sum(eigenvals)
    eigenpairs = [(eigenvals[i], eigenvects[:,i], i) for i in range(eigenvals.shape[0])]
    eigenpairs.sort(key=lambda x: x[0], reverse=True)

    ev = 0
    i = 0
    eigenvects = []
    idxs = []
    while ev < explained_var:
        ev += eigenpairs[i][0]
        eigenvects.append(eigenpairs[i][1])
        idxs.append(eigenpairs[i][2])

    # return new dataset and the ids of the most relevant attributes
    return np.transpose(eigenvects), idxs

def skpca(dataset, explained_var):
    pca = PCA(n_components=explained_var)
    pca.fit(dataset)
    return pca.components_.T, pca.explained_variance_ratio_
