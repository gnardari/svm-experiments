import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def pca(dataset, explained_var):

    if dataset.dtype == 'sparse':
        dataset = dataset.todense()

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

def biplot(score, y, coeff, labels=None):
    xs = score[:,0]
    ys = score[:,1]
    n = coeff.shape[0]
    scalex = 1.0/(xs.max() - xs.min())
    scaley = 1.0/(ys.max() - ys.min())
    plt.scatter(xs * scalex,ys * scaley, c = y)
    for i in range(n):
        plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'r',alpha = 0.5)
        if labels is None:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, "Var"+str(i+1), color = 'g', ha = 'center', va = 'center')
        else:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color = 'g', ha = 'center', va = 'center')

    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.xlabel("PC{}".format(1))
    plt.ylabel("PC{}".format(2))
    plt.grid()

def analyse_pca(x, y, eigenv, explained_var, k=5):
    eigenv = eigenv[:k]
    explained_var = explained_var[:k]

    print('top {} pca attributes: {}, explained var: {}'.format(
        k, eigenv, np.sum(explained_var)
        ))

    np.random.shuffle(x)
    biplot(x[:,0:2], y, np.transpose(eigenv[0:2, :]))
    plt.show()

    # plot quais os atributos do pca mais relevantes ?
    # quais features
    # print('einvects: {}'.format(eigenvects.shape))
    # print('Explained var: {}'.format(explained_var))
