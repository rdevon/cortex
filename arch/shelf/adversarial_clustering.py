'''Adversarial clustering.

'''

import itertools

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn import metrics
import torch

from ali import build_discriminator as build_mine_discriminator, build_extra_networks, score, apply_penalty
from featnet import (apply_gradient_penalty, build_encoder, build_discriminator as build_noise_discriminator, encode,
                     get_results, score as featnet_score, visualize)
from minet import network_routine
from gan import generator_loss
from utils import update_decoder_args, update_encoder_args, to_one_hot


def encoder_routine(data, models, losses, results, viz, mine_measure=None, noise_measure=None,
                    encoder_penalty_amount=0., generator_loss_type='non-saturating', beta=0.):
    X_P, X_Q, T, Z_P = data.get_batch('1.images', '2.images', '1.targets', 'y')
    encoder = models.encoder
    dim_l = data.get_dims('labels')

    Z = encoder(X_P, nonlinearity=torch.nn.Softmax(dim=1))
    C = Z.argmax(1).data.cpu().numpy()
    T_ = T
    T = T.data.cpu().numpy()
    ARI = metrics.adjusted_rand_score(T, C)
    AMI = metrics.adjusted_mutual_info_score(T, C)
    homogeneity = metrics.homogeneity_score(T, C)
    completeness = metrics.completeness_score(T, C)
    v_measure = metrics.v_measure_score(T, C)
    FMI = metrics.fowlkes_mallows_score(T, C)

    C_locs = (C[:, None] == np.arange(dim_l)).astype('float')
    T_locs = (T[:, None] == np.arange(dim_l)).astype('float')

    loss = (C_locs[:, :, None] != T_locs[:, None, :]).mean(0)
    rows, cols = linear_sum_assignment(loss)

    acc = (C_locs[:, rows].argmax(1) == T_locs[:, cols].argmax(1)).mean()

    results.update(Cluster_scores=dict(ARI=ARI, AMI=AMI, homogeneity=homogeneity, completeness=completeness,
                                       v_measure=v_measure, FMI=FMI, ACC=acc))

    E_pos_n, E_neg_n, P_samples_n, Q_samples_n = featnet_score(models, Z_P, Z, noise_measure,
                                                               key='noise_discriminator')
    get_results(P_samples_n, Q_samples_n, E_pos_n, E_neg_n, noise_measure, results=results, name='noise')
    losses.encoder = generator_loss(Q_samples_n, noise_measure, loss_type=generator_loss_type)

    E_pos, E_neg, P_samples, Q_samples = score(models, X_P, X_Q, Z, Z, mine_measure, key='mine_discriminator')

    losses.encoder += beta * (E_neg - E_pos)
    get_results(P_samples, Q_samples, E_pos, E_neg, mine_measure, results=results, name='mine')
    visualize(Z, P_samples, Q_samples, X_P, T_, viz=viz, scatter=True)

    penalty = apply_gradient_penalty(data, models, inputs=X_P, model='encoder', penalty_amount=encoder_penalty_amount)
    if penalty:
        losses.encoder = penalty

    #data.noise['y'][0].concentration *= 0.9999
    #data.noise['y'][1].concentration *= 0.9999
    results.update(alpha=data.noise['y'][0].concentration.mean().item())


def mine_discriminator_routine(data, models, losses, results, viz, mine_measure='JSD', mine_penalty_amount=0.5):
    X_P, X_Q, T = data.get_batch('1.images', '2.images', '1.targets')
    encoder = models.encoder
    Z = encoder(X_P, nonlinearity=torch.nn.Softmax(dim=1))

    E_pos, E_neg, _, _ = score(models, X_P, X_Q, Z, Z, mine_measure, key='mine_discriminator')

    losses.mine_discriminator = E_neg - E_pos
    penalty = apply_penalty(models, losses, results, X_P, Z, mine_penalty_amount, key='mine_discriminator')

    if penalty:
        losses.mine_discriminator += penalty


def noise_discriminator_routine(data, models, losses, results, viz, noise_penalty_amount=0.5, noise_measure='JSD'):
    X, Z_P = data.get_batch('1.images', 'y')
    encoder = models.encoder

    Z_Q = encoder(X, nonlinearity=torch.nn.Softmax(dim=1))
    E_pos, E_neg, _, _ = featnet_score(models, Z_P, Z_Q, noise_measure, key='noise_discriminator')
    losses.noise_discriminator = E_neg - E_pos

    penalty = apply_gradient_penalty(data, models, inputs=(Z_P, Z_Q), model='noise_discriminator',
                                     penalty_amount=noise_penalty_amount)
    if penalty:
        losses.noise_discriminator += penalty


def BUILD(data, models, encoder_type='convnet', decoder_type='convnet', encoder_args={}, decoder_args={},
          noise_parameters=dict(concentration=0.1), noise_type='dirichlet'):

    x_shape = data.get_dims('x', 'y', 'c')
    dim_l = data.get_dims('labels')

    data.add_noise('y', dist=noise_type, size=dim_l, **noise_parameters)

    Encoder, encoder_args = update_encoder_args(x_shape, model_type=encoder_type, encoder_args=encoder_args)
    Decoder, decoder_args = update_decoder_args(x_shape, model_type=decoder_type, decoder_args=decoder_args)
    build_mine_discriminator(models, x_shape, dim_l, Encoder, key='mine_discriminator', **encoder_args)
    build_noise_discriminator(models, dim_l, key='noise_discriminator')
    build_encoder(models, x_shape, dim_l, Encoder, **encoder_args)
    build_extra_networks(models, x_shape, dim_l, dim_l, Decoder, **decoder_args)


TRAIN_ROUTINES = dict(mine_discriminator=mine_discriminator_routine, noise_discriminator=noise_discriminator_routine,
                      encoder=encoder_routine, nets=network_routine)

DEFAULT_CONFIG = dict(data=dict(batch_size=dict(train=64, test=640), duplicate=2))