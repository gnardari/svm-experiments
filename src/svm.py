from sys import maxint
from itertools import combinations, imap
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import polynomial_kernel
from sklearn.metrics.pairwise import rbf_kernel
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
    clf = GridSearchCV(SVC(cache_size=12000), param_grid, cv=10,
                       scoring='accuracy',
                       n_jobs=4, verbose=1)
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
    accuracy = [1 if x == y else 0
                for x,y in zip(y_true, y_pred)]
    accuracy = np.sum(accuracy) / len(accuracy)
    print(classification_report(y_true, y_pred))

    return clf.best_estimator_, clf.best_params_, accuracy

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
        gram = rbf_kernel(X, Y, gamma)
    return gram

def calc_radius(gram):
    # matriz simetrica, pegar somente metade
    g = np.hstack([np.diagonal(gram, offset=i)
                   for i in range(gram.shape[0])])
    g = np.random.choice(g, int(0.01 * g.shape[0]), replace=False)
    # comparar todos os itens da matriz de gram entre si
    dist = lambda uv: np.linalg.norm(uv[0]-uv[1])**2
    max_dist = max(imap(dist, combinations(g,2)))
    return np.sqrt(max_dist)/2

def calc_margin(sgram, dual_coefs):
    margins = []
    for alphas in dual_coefs:
        alphas = np.outer(alphas, alphas)
        W = alphas * sgram
        margins.append(1 / np.sqrt(np.sum(W ** 2)))
    return min(margins)

def calc_vc_dim(r, rho):
    return (r/rho)**2

def divergence_factor(n, delta, vc):
    return np.sqrt(4.0/n *
                   (vc *
                    np.log(n)**2 *
                    np.log(1.0/delta)))

def examples_inside_margin(gram, dual_coefs, intercepts):
    nus = []
    for i, alphas in enumerate(dual_coefs):
        out_sum = np.outer(alphas, alphas).sum(axis=1)
        nus.append(((out_sum - intercepts[i]) < 1).sum()
                     / float(gram.shape[0]))
    return max(nus)

def solve_gen_bound_for_n(vc, delta=0.05):
    res = 10
    n = 1000
    while n < 10**10 and res > 0.05:
        res = divergence_factor(n, delta, vc)
        n += 1000
    return n

def generalization_bound(gram, clf, delta=0.05):
    dual_coefs = clf.dual_coef_
    svs = clf.support_
    intercepts =  clf.intercept_
    # support vectors in the gram matrix
    sgram = gram[svs][:,svs]
    n = gram.shape[0]
    radius = calc_radius(gram)
    margin = calc_margin(sgram, dual_coefs)
    vc = calc_vc_dim(radius, margin)
    df = divergence_factor(n, delta, vc)
    nu = examples_inside_margin(gram, dual_coefs, intercepts)
    gb = nu + df
    n_gb = solve_gen_bound_for_n(vc)

    return n, radius, margin, vc, gb, n_gb, nu

def train_svm(X_train, y_train, params):
    gram = calc_kernel(X_train, params=params)
    clf = SVC(kernel='precomputed', cache_size=1200)
    clf.fit(gram, y_train)
    return gram, clf
