'''Toy dataset clustering

'''

import itertools
import numpy as np
from scipy.special import digamma, gamma
from sklearn import metrics
import torch

from ali import apply_penalty, score
from featnet import score as featnet_score, get_results
from gan import generator_loss, apply_gradient_penalty
from modules.fully_connected import FullyConnectedNet
from utils import to_one_hot


def inverse_digamma(X, iterations=3):
    L = 1
    Y = np.exp(X)
    while L > 10e-8:
        Y = Y + L * np.sign(X - digamma(Y))
        L /= 2
    return Y


def fixed_point_alpha(alpha, log_p_hat):
    alpha_new = inverse_digamma(digamma(sum(alpha)) + log_p_hat)
    return alpha_new


# Must have data, models, losses, results, and viz. **kargs
# should match the keys in DEFAULT_CONFIG.routines below.
def noise_discriminator_routine(data, models, losses, results, viz, noise_measure='JSD',
                                alpha_lr=0.0005,
                                learn_alpha=False):
    X, Y_P = data.get_batch('1.images', 'y')
    encoder = models.encoder
    Y_Q = encoder(X, nonlinearity=torch.nn.Softmax(dim=1))

    log_p_hat = torch.log(Y_Q).mean(0)
    alpha = data.noise['y'][0].concentration[0]
    alpha_permutations = np.array(list(itertools.permutations(alpha)))
    # assert False, np.array(alpha_permutations).shape

    log_likelihoods = (np.log(gamma(alpha_permutations.sum(1))) -
                       np.log(gamma(alpha_permutations)).sum(1) +
                       ((alpha_permutations - 1.) * log_p_hat.data.cpu().numpy()[None, :]).sum(1))
    idx = log_likelihoods.argmax()
    new_alpha = alpha_permutations[idx]
    data.noise['y'][0].concentration = (
        data.noise['y'][0].concentration * 0 + torch.tensor(new_alpha))
    data.noise['y'][1].concentration = (
        data.noise['y'][1].concentration * 0 + torch.tensor(new_alpha))
    results.update(Switched_alpha=sum(alpha.data.data.cpu().numpy() != new_alpha))

    if learn_alpha:
        alpha = data.noise['y'][0].concentration[1]
        results.update(dict(Alphas=dict((str(i), alpha) for i, alpha in enumerate(alpha))))
        log_p_hat = torch.log(Y_Q).mean(0)

        '''
        g = (digamma(sum(alpha)) - digamma(sum(alpha))).cuda() + log_p_hat
        data.noise['y'][0].concentration += alpha_lr * g.data.cpu()
        data.noise['y'][1].concentration += alpha_lr * g.data.cpu()
        '''

        new_alpha = torch.tensor(fixed_point_alpha(alpha.numpy(), log_p_hat.data.cpu().numpy()))
        beta = 0.0001
        data.noise['y'][0].concentration = ((1 - beta) *
                                            data.noise['y'][0].concentration +
                                            beta * new_alpha)
        data.noise['y'][1].concentration = ((1 - beta) *
                                            torch.zeros_like(data.noise['y'][1].concentration) +
                                            beta * new_alpha)

    E_pos, E_neg, _, _ = featnet_score(models, Y_P, Y_Q, noise_measure, key='noise_discriminator')
    losses.noise_discriminator = E_neg - E_pos


def mine_discriminator_routine(data, models, losses, results, viz, mine_measure='JSD'):
    X_P, X_Q, T = data.get_batch('1.images', '2.images', '1.targets')
    encoder = models.encoder

    Y_Q = encoder(X_P, nonlinearity=torch.nn.Softmax(dim=1))
    E_pos, E_neg, P_samples, Q_samples = score(models, X_P, X_Q, Y_Q, Y_Q, mine_measure, key='mine')

    losses.mine = E_neg - E_pos


def encoder_routine(data, models, losses, results, viz, mine_measure=None, noise_measure=None,
                    generator_loss_type='non-saturating', beta=5.0):
    X_P, X_Q, T, Y_P = data.get_batch('1.images', '2.images', '1.targets', 'y')
    dim_l = data.get_dims('labels')
    encoder = models.encoder
    Y_Q = encoder(X_P, nonlinearity=torch.nn.Softmax(dim=1))

    D = Y_Q.argmax(1)
    C = D.data.cpu().numpy()
    ARI = metrics.adjusted_rand_score(T, C)
    AMI = metrics.adjusted_mutual_info_score(T, C)
    homogeneity = metrics.homogeneity_score(T, C)
    completeness = metrics.completeness_score(T, C)
    v_measure = metrics.v_measure_score(T, C)
    FMI = metrics.fowlkes_mallows_score(T, C)
    results.update(Cluster_scores=dict(ARI=ARI, AMI=AMI, homogeneity=homogeneity,
                                       completeness=completeness,
                                       v_measure=v_measure, FMI=FMI))
    class_numbers = {}
    target_numbers = to_one_hot(T, dim_l).sum(0)
    found_numbers = to_one_hot(D, dim_l).sum(0)
    for l in range(dim_l):
        class_numbers['GT_{}'.format(l)] = target_numbers[l]
        class_numbers['EN_{}'.format(l)] = found_numbers[l]
    results.update(Class_numbers=class_numbers)

    # Featnet
    E_pos_n, E_neg_n, P_samples_n, Q_samples_n = featnet_score(models, Y_P, Y_Q, noise_measure,
                                                               key='noise_discriminator')
    get_results(P_samples_n, Q_samples_n, E_pos_n, E_neg_n,
                noise_measure, results=results, name='noise')
    losses.encoder = generator_loss(Q_samples_n, noise_measure, loss_type=generator_loss_type)

    # MINE
    E_pos, E_neg, P_samples, Q_samples = score(models, X_P, X_Q, Y_Q, Y_Q, mine_measure, key='mine')

    losses.encoder += beta * (E_neg - E_pos)
    get_results(P_samples, Q_samples, E_pos, E_neg, mine_measure, results=results, name='mine')

    viz.add_scatter(X_P, labels=D, name='Clusters')
    viz.add_histogram(dict(real=P_samples_n.view(-1).data,
                      fake=Q_samples_n.view(-1).data), name='discriminator output')


def penalty_routine(data, models, losses, results, viz, mine_penalty_amount=0.2, penalty_amount=0.2,
                    encoder_penalty_amount=1.0):
    X_P, X_Q, Y_P = data.get_batch('1.images', '2.images', 'y')
    encoder = models.encoder
    Y_Q = encoder(X_P, nonlinearity=torch.nn.Softmax(dim=1))

    penalty = apply_gradient_penalty(data, models, inputs=(Y_P, Y_Q), model='noise_discriminator',
                                     penalty_amount=penalty_amount)
    if penalty:
        losses.noise_discriminator = penalty

    penalty = apply_penalty(models, losses, results, X_P, Y_Q, mine_penalty_amount, key='mine')
    if penalty:
        losses.mine = penalty

    penalty = apply_gradient_penalty(data, models, inputs=X_P, model='encoder',
                                     penalty_amount=encoder_penalty_amount)
    if penalty:
        losses.encoder = penalty


# CORTEX ===================================================================================
# Must include `BUILD` and `TRAIN_ROUTINES`

def BUILD(data, models, noise_type='dirichlet', noise_parameters=dict(concentration=0.1),
          encoder_args=dict(dim_h=[100, 100], batch_norm=True),
          noise_discriminator_args=dict(dim_h=[200, 10], batch_norm=False),
          mine_discriminator_args=dict(dim_h=[200, 10], batch_norm=False)):
    dim_in, dim_l = data.get_dims('x', 'labels')
    data.reset(make_pbar=False, test=True)
    data.next()
    T = data.get_batch('1.targets')

    target_numbers = to_one_hot(T, dim_l).sum(0)
    N = sum(target_numbers)
    target_freqencies = [float(n) / float(N) for n in target_numbers]

    '''
    target_numbers = to_one_hot(T, dim_l).sum(0)
    N = sum(target_numbers)
    target_freqencies = [float(n) / float(N) for n in target_numbers]
    p1, p2 = target_freqencies
    m = T.mean().item()
    v = (1. / float(N)) * ((T - m) ** 2).sum().item() * 0.95
    alpha = m * ((m * (1 - m) / v) - 1)
    beta = (1 - m) * ((m * (1 - m) / v) - 1)
    '''

    alpha = np.array([1. for _ in range(dim_l)])

    b = 0.0001
    log_p_hat = torch.log((1. - b) * to_one_hot(T, dim_l) + b).mean(0)

    for _ in range(10):
        print(alpha)
        alpha = torch.tensor(fixed_point_alpha(alpha, log_p_hat.data.cpu().numpy())).data.numpy()
    logger.info('Found starting alphas: {}'.format(alpha))
    print(target_freqencies)
    # assert False, alpha

    noise_parameters.update(concentration=alpha.tolist())
    # noise_parameters.update(concentration=target_freqencies)
    # assert False, noise_parameters
    data.add_noise('y', dist=noise_type, size=dim_l, **noise_parameters)

    encoder = FullyConnectedNet(dim_in, dim_out=dim_l, **encoder_args)
    mine_bot = FullyConnectedNet(dim_in, dim_out=dim_l, **mine_discriminator_args)
    mine_top = FullyConnectedNet(dim_l, dim_out=dim_l, **mine_discriminator_args)
    mine_fin = FullyConnectedNet(2 * dim_l, dim_out=1, **mine_discriminator_args)

    noise_discriminator = FullyConnectedNet(dim_l, dim_out=1, **noise_discriminator_args)

    models.update(mine=(mine_bot, mine_top, mine_fin), noise_discriminator=noise_discriminator,
                  encoder=encoder)


# Dictionary reference to train routines. Keys are up to you
TRAIN_ROUTINES = dict(mine=mine_discriminator_routine, discriminator=noise_discriminator_routine,
                      encoder=encoder_routine, penalty=penalty_routine)

# Dictionary reference to test routines. If not set, will be copied from train. If value is None,
# will not be used in test.
TEST_ROUTINES = dict(penalty=None)

# Default configuration for this model
DEFAULT_CONFIG = dict(
    data=dict(batch_size=dict(train=64, test=1000), duplicate=2),
    optimizer=dict(),
    train=dict(epochs=1000)
)
