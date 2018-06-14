'''Module for handling noise.

'''

import torch
import torch.distributions as tdist


_dist_dict = dict(
    bernoulli=tdist.bernoulli.Bernoulli,
    beta=tdist.beta.Beta,
    binomial=tdist.binomial.Binomial,
    categorical=tdist.categorical.Categorical,
    cauchy=tdist.cauchy.Cauchy,
    chi2=tdist.chi2.Chi2,
    dirichlet=tdist.dirichlet.Dirichlet,
    exponential=tdist.exponential.Exponential,
    fishersnedecor=tdist.fishersnedecor.FisherSnedecor,
    gamma=tdist.gamma.Gamma,
    geometric=tdist.geometric.Geometric,
    gumbel=tdist.gumbel.Gumbel,
    laplace=tdist.laplace.Laplace,
    log_normal=tdist.log_normal.LogNormal,
    multinomial=tdist.multinomial.Multinomial,
    multivariate_normal=tdist.multivariate_normal.MultivariateNormal,
    normal=tdist.normal.Normal,
    one_hot_categorical=tdist.one_hot_categorical.OneHotCategorical,
    pareto=tdist.pareto.Pareto,
    poisson=tdist.poisson.Poisson,
    relaxed_bernoulli=tdist.relaxed_bernoulli.RelaxedBernoulli,
    relaxed_categorical=tdist.relaxed_categorical.RelaxedOneHotCategorical,
    studentT=tdist.studentT.StudentT,
    uniform=tdist.uniform.Uniform
)


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

    Dist = _dist_dict.get(dist)
    if Dist is None:
        raise NotImplementedError(dist)

    if dist == 'dirichlet':
        conc = kwargs.pop('concentration', 1.)
        conc = expand(conc)
        var = Dist(conc, **kwargs)
    elif dist in ('cachy', 'gumbel', 'laplace', 'log_normal', 'normal'):
        loc = kwargs.pop('loc', 0.)
        scale = kwargs.pop('scale', 1.)
        loc, scale = expand(loc, scale)
        var = Dist(loc, scale, **kwargs)
    elif dist == 'uniform':
        low = kwargs.pop('low', 0.)
        high = kwargs.pop('high', 1.)
        low, high = expand(low, high)
        var = Dist(low, high, **kwargs)
    else:
        raise NotImplementedError('`{}` distribution not found'.format(dist))

    return var
