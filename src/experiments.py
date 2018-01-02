from circle import find_radius
from read_datasets import *
from feature_extractors import *
from svm import *
from pca import skpca
import numpy as np
import cPickle

def subsample(dataset, labels, sample_pct=0.5):
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

def mnist_experiment():
    mnist = read_mnist(test=True)
    X_train, y_train = subsample(mnist['train']['data'],
                                 mnist['train']['labels'],
                                 0.05)

    X_test, y_test = subsample(mnist['test']['data'],
                               mnist['test']['labels'],
                               0.1)

    del mnist

    print('Applying feature extractors')
    # aplicando filtro nas imagens
    X_train = hog(X_train)
    X_test = hog(X_test)

    print('Running PCA')
    # # reduzindo dimensao com PCA
    eigenvects, explained_var = skpca(X_train, 0.99)

    X_train = normalize(np.dot(X_train, eigenvects))
    X_test = normalize(np.dot(X_test, eigenvects))

    # print('Training set size: %d, %d' % X_train.shape)
    # print('Test set size: %d, %d' % X_test.shape)
    #
    # clf, params = grid_search_svm(X_train, y_train,
    #                       X_test, y_test, 'mnist')

    # with open('../models/mnist.pkl', 'wb') as f:
    #      cPickle.dump(svm, f)

    params = {}
    gram, clf = train_svm(X_train, y_train,
                          X_test, y_test, params)

    res = generalization_bound(gram, clf.dual_coef_, clf.support_)
    return gram, clf, res

def imagenet_experiment():
    imagenet = read_imagenet(test=True)

    X_train, y_train = (imagenet['train']['data'],
                       imagenet['train']['labels'])

    X_test, y_test = (imagenet['test']['data'],
                      imagenet['test']['labels'])

    del imagenet

    print('Applying feature extractors')
    # # aplicando filtro nas imagens
    X_train = vgg_convolutions(X_train)
    X_test = vgg_convolutions(X_test)

    print('Running PCA')
    # # reduzindo dimensao com PCA
    eigenvects = skpca(X_train, 0.99)
    X_train = normalize(np.dot(X_train, eigenvects))
    X_test = normalize(np.dot(X_test, eigenvects))

    print('Training set size: %d, %d' % X_train.shape)
    print('Test set size: %d, %d' % X_test.shape)

    svm = grid_search_svm(X_train, y_train,
                          X_test, y_test, 'imagenet')

    with open('../models/imagenet.pkl', 'wb') as f:
         cPickle.dump(svm, f)

def msd_experiment():
    msd = read_msd(test=True)

    X_train, y_train = (msd['train']['data'],
                       msd['train']['labels'])

    X_test, y_test = (msd['test']['data'],
                      msd['test']['labels'])

    del msd

    X_train, y_train = subsample(X_train, y_train, 0.50)
    X_test, y_test = subsample(X_test, y_test, 1)

    print('Running PCA')
    # reduzindo dimensao com PCA
    # eigenvects = skpca(X_train, 0.98)
    # X_train = normalize(np.dot(X_train, eigenvects))
    # X_test = normalize(np.dot(X_test, eigenvects))

    print('Training set size: %d, %d' % X_train.shape)
    print('Test set size: %d, %d' % X_test.shape)

    svm = grid_search_svm(X_train, y_train,
                          X_test, y_test, 'msd')

    with open('../models/msd.pkl', 'wb') as f:
         cPickle.dump(svm, f)

def movie_review_experiment():
    reviews = read_movie_reviews()

    X_train, y_train = (reviews['train']['data'],
                       reviews['train']['labels'])

    X_test, y_test = (reviews['test']['data'],
                      reviews['test']['labels'])

    del reviews

    print('Applying feature extractors')
    X_train, X_test = tfidf(X_train, X_test)

    print('Training set size: %d, %d' % X_train.shape)
    print('Test set size: %d, %d' % X_test.shape)
    svm = grid_search_svm(X_train, y_train,
                          X_test, y_test, 'reviews')

    with open('../models/reviews.pkl', 'wb') as f:
         cPickle.dump(svm, f)

# movie_review_experiment()
gram, clf, res = mnist_experiment()
print(res)
# with open('../models/mnist.pkl', 'rb') as f:
#      svm = cPickle.load(f)
