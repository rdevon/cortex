'''Module for handling noise.

'''

import torch


def get_noise_var(dist, size, **kwargs):

    def expand(*args):
        expanded = tuple()
        for arg in args:
            zeros = torch.zeros(size)
            if isinstance(arg, list):
                arg_tensor = torch.tensor(arg)
                expanded += (zeros + arg_tensor,)
            else:
                expanded += (zeros + arg,)
        return expanded

    if dist == 'bernoulli':
        Dist = torch.distributions.bernoulli.Bernoulli
        raise NotImplementedError(dist)
    elif dist == 'beta':
        Dist = torch.distributions.beta.Beta
        raise NotImplementedError(dist)
    elif dist == 'binomial':
        Dist = torch.distributions.binomial.Binomial
        raise NotImplementedError(dist)
    elif dist == 'categorical':
        Dist = torch.distributions.categorical.Categorical
        raise NotImplementedError(dist)
    elif dist == 'cauchy':
        Dist = torch.distributions.cauchy.Cauchy
        raise NotImplementedError(dist)
    elif dist == 'chi2':
        Dist = torch.distributions.chi2.Chi2
        raise NotImplementedError(dist)
    elif dist == 'dirichlet':
        Dist = torch.distributions.dirichlet.Dirichlet
        conc = kwargs.pop('concentration', 1.)
        conc = expand(conc)
        var = Dist(conc, **kwargs)
    elif dist == 'exponential':
        Dist = torch.distributions.exponential.Exponential
        raise NotImplementedError(dist)
    elif dist == 'fishersnedecor':
        Dist = torch.distributions.fishersnedecor.FisherSnedecor
        raise NotImplementedError(dist)
    elif dist == 'gamma':
        Dist = torch.distributions.gamma.Gamma
        raise NotImplementedError(dist)
    elif dist == 'geometric':
        Dist = torch.distributions.geometric.Geometric
        raise NotImplementedError(dist)
    elif dist == 'multinomial':
        Dist = torch.distributions.multinomial.Multinomial
        raise NotImplementedError(dist)
    elif dist == 'multivariate_normal':
        Dist = torch.distributions.multivariate_normal.MultivariateNormal
        raise NotImplementedError(dist)
    elif dist in ('cachy', 'gumbel', 'laplace', 'log_normal', 'normal'):
        Dist = torch.distributions.normal.Normal
        loc = kwargs.pop('loc', 0.)
        scale = kwargs.pop('scale', 1.)
        loc, scale = expand(loc, scale)
        var = Dist(loc, scale, **kwargs)
    elif dist == 'one_hot_categorical':
        Dist = torch.distributions.one_hot_categorical.OneHotCategorical
        raise NotImplementedError(dist)
    elif dist == 'pareto':
        Dist = torch.distributions.pareto.Pareto
        raise NotImplementedError(dist)
    elif dist == 'poisson':
        Dist = torch.distributions.poisson.Poisson
        raise NotImplementedError(dist)
    elif dist == 'relaxed_bernoulli':
        Dist = torch.distributions.relaxed_bernoulli.RelaxedBernoulli
        raise NotImplementedError(dist)
    elif dist == 'relaxed_categorical':
        Dist = torch.distributions.relaxed_categorical.RelaxedOneHotCategorical
        raise NotImplementedError(dist)
    elif dist == 'studentT':
        Dist = torch.distributions.studentT.StudentT
        raise NotImplementedError(dist)
    elif dist == 'uniform':
        Dist = torch.distributions.uniform.Uniform
        low = kwargs.pop('low', 0.)
        high = kwargs.pop('high', 1.)
        low, high = expand(low, high)
        var = Dist(low, high, **kwargs)
    else:
        raise NotImplementedError('`{}` distribution not found'.format(dist))

    d_args = dict(
        beta=['concentration1', 'concentration0'],
        cachy=['loc', 'scale'],
        chi2=['df'],
        dirichlet=['concentration'],
        exponential=['rate'],
        fishersnedecor=['df1', 'df2'],
        gamma=['concentration', 'rate'],
        gumbel=['loc', 'scale'],
        laplace=['loc', 'scale'],
        log_normal=['loc', 'scale'],
        multivariate_normal=['loc'],
        normal=['loc', 'scale'],
        pareto=['scale', 'alpha'],
        poisson=['rate'],
        relaxed_bernoulli=['temperature'],
        relaxed_categorical=['temperature'],
        studentT=['df'],
        uniform=['high', 'low']
    )

    return var
