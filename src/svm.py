from itertools import combinations
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import polynomial_kernel
from sklearn.gaussian_process.kernels import RBF
import numpy as np

def grid_search_svm(X_train, y_train, X_test, y_test, dataset_name):
    param_grid = [{
        'kernel': ['rbf'],
        'gamma': [0.1, 0.01, 1e-3, 1e-4]
    }, {
        'kernel': ['poly'],
        'gamma': [0.1, 0.01, 1e-3, 1e-4],
        'coef0': [0, 1, 2],
        'degree': [2, 3, 4, 5]
    },{
        'kernel': ['linear']
    }]

    print("# Tuning hyper-parameters for the %s dataset" % dataset_name)
    clf = GridSearchCV(SVC(cache_size=800), param_grid, cv=10, n_jobs=4, verbose=100)
    clf.fit(X_train, y_train)

    print("Grid scores on development set:")
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))

    print("Best parameters set found on development set:")
    print(clf.best_params_)

    print("Detailed classification report:")
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")

    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))

    return clf.best_estimator_, clf.best_params_

def calc_kernel(X, Y=None, params={}):
    kernel = params.get('kernel', 'linear')
    degree = params.get('degree', 1)
    gamma = params.get('gamma', 1)
    coef0 = params.get('coef0', 0)

    if kernel == 'linear':
        gram = polynomial_kernel(X, Y,
                                 degree=1, coef0=0)
    elif kernel == 'polynomial':
        gram = polynomial_kernel(X, Y,
                                 degree=degree,
                                 gamma=gamma,
                                 coef0=coef0)
    else:
        # verificar se lenth_scale eh realmente igual a gamma
        gram = RBF(X, Y, length_scale=gamma)
    return gram

def calc_radius(gram):
    # pegar somente metade da matriz,
    # ja que dist(i,j) = dist(j,i)
    g = np.hstack([np.diagonal(gram, offset=i)
                   for i in range(gram.shape[0])])
    # comparar todos os itens da matriz de gram entre si
    dist = lambda uv: np.linalg.norm(uv[0]-uv[1])**2
    max_dist = max(map(dist, combinations(g,2)))
    return np.sqrt(max_dist)/2

def calc_margin(sgram, dual_coef):
    margins = []
    for class_cd in dual_coef:
        class_cd = np.outer(class_cd, class_cd)
        W = class_cd * sgram
        margins.append(1 / np.sqrt(np.sum(W ** 2)))
    return max(margins)

def shattering(r, rho):
    return (r/rho)**2

def solve_gen_bound_for_n(r, rho, delta=0.05):
    return np.exp(-(rho*np.sqrt(delta))/r)

def generalization_bound(gram, dual_coefs, svs, delta=0.05):
    # support vectors in the gram matrix
    sgram = gram[svs][:,svs]
    n = gram.shape[0]
    radius = calc_radius(gram)
    margin = calc_margin(sgram, dual_coefs)
    stt = shattering(radius, margin)

    gb = np.sqrt(4.0/n *
                stt *
                np.log(n)**2 *
                np.log(1.0/delta))

    n_gb = solve_gen_bound_for_n(radius, margin)

    return n, radius, margin, stt, gb, n_gb

def train_svm(X_train, y_train, X_test, y_test, params):
    gram = calc_kernel(X_train, params=params)

    clf = SVC(kernel='precomputed', cache_size=800)
    clf.fit(gram, y_train)

    X_test = calc_kernel(X_test, X_train, params)
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))

    return gram, clf

