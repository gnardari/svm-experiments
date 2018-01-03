from circle import find_radius
from read_datasets import *
from feature_extractors import *
from svm import *
from pca import *
import numpy as np
import cPickle

def subsample(dataset, labels, sample_pct=1.0):

    if sample_pct == 1.0:
        return dataset, labels

    dataset_size = len(dataset)
    idx = np.random.choice(dataset_size,
                           int(dataset_size*sample_pct),
                           replace=False)
    dataset = dataset[idx]
    labels = labels[idx]
    return dataset, labels

def normalize(d):
    d /= np.max(np.abs(d),axis=0)
    return d


datasets = {
    'mnist': read_mnist,
    'imagenet': read_imagenet,
    'msd': read_msd,
    'movies': read_movie_reviews
}

feature_extractors = {
    'vgg': vgg_convolutions,
    'hog': hog,
    'haralick': haralick,
    'tfidf': tfidf
}

def experiment(dataset_name, sample_pct, extractors, pca=0.99):
    dataset = datasets[dataset_name]()
    X_train, y_train = subsample(dataset['train']['data'],
                                 dataset['train']['labels'],
                                 sample_pct)

    X_test, y_test = subsample(dataset['test']['data'],
                               dataset['test']['labels'])
    del dataset

    for e in extractors:
        if e == 'tfidf':
            X_train, X_test = feature_extractors[e](X_train, X_test)
        else:
            X_train = feature_extractors[e](X_train)
            X_test = feature_extractors[e](X_test)

    if pca:
        eigenvects, explained_var = skpca(X_train, pca)
        X_train = normalize(np.dot(X_train, eigenvects))
        X_test = normalize(np.dot(X_test, eigenvects))

        analyse_pca(X_train, y_train, eigenvects, explained_var)

    print('Training set size: %d, %d' % X_train.shape)
    print('Test set size: %d, %d' % X_test.shape)

    # clf, params = grid_search_svm(X_train, y_train,
    #                       X_test, y_test, dataset_name)
    #
    # with open('../models/{}.pkl'.format(dataset_name), 'wb') as f:
    #      cPickle.dump(clf, f)

    params = {}
    gram, clf = train_svm(X_train, y_train,
                          X_test, y_test, params)

    res = generalization(gram, clf.dual_coef_, clf.support_)
    return gram, clf, res


experiment('mnist', 0.05, ['hog'])
# experiment('imagenet', 1.0, ['vgg'])
# experiment('msd', 0.1, [], pca=False)
# experiment('movies', 0.5, ['tfidf'], pca=2)
