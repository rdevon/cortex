'''Model misc utilities.

'''

import torch
from torch import nn


def cross_correlation(X, remove_diagonal=False):
    X_s = X / X.std(0)
    X_m = X_s - X_s.mean(0)
    b, dim = X_m.size()
    correlations = (X_m.unsqueeze(2).expand(b, dim, dim) * X_m.unsqueeze(1).expand(b, dim, dim)).sum(0) / float(b)
    if remove_diagonal:
        Id = torch.Tensor(dim, dim)
        nn.init.eye(Id)
        Id = torch.autograd.Variable(Id.cuda(), requires_grad=False)
        correlations -= Id

    return correlations