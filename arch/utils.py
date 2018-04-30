'''Model misc utilities.

'''

import logging

logger = logging.getLogger('cortex.arch' + __name__)

try:
    import thundersvmScikit as svm
    use_thundersvm = True
except ImportError:
    from sklearn import svm
    logger.warning('Using sklearn SVM. This will be SLOW. Install thundersvm and add to your PYTHONPATH')
    use_thundersvm = False
import torch


def cross_correlation(X, remove_diagonal=False):
    X_s = X / X.std(0)
    X_m = X_s - X_s.mean(0)
    b, dim = X_m.size()
    correlations = (X_m.unsqueeze(2).expand(b, dim, dim) * X_m.unsqueeze(1).expand(b, dim, dim)).sum(0) / float(b)
    if remove_diagonal:
        Id = torch.eye(dim)
        Id = torch.autograd.Variable(Id.cuda(), requires_grad=False)
        correlations -= Id

    return correlations


def perform_svc(X, Y, clf=None):
    if clf is None:
        if use_thundersvm:
            clf = svm.SVC(kernel=0, verbose=True)
        else:
            clf = svm.LinearSVC()
        clf.fit(X, Y)

    Y_hat = clf.predict(X)

    return clf, Y_hat