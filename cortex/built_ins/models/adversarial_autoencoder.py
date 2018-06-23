# flake8: noqa
'''Adversarial autoencoder
'''

import torch
import torch.nn.functional as F

from classifier import classify
from gan import generator_loss
from featnet import (apply_gradient_penalty, build_discriminator, get_results,
                     score, shape_noise, visualize)
from modules.fully_connected import FullyConnectedNet
# from utils import cross_correlation
from vae import build_encoder, build_decoder
from utils import update_decoder_args, update_encoder_args


def encode(models,
           X,
           output_nonlin=False,
           noise_type='hypercubes',
           key='autoencoder'):
    encoder = models[key][0]

    Z_Q = encoder(X)
    if output_nonlin:
        if noise_type == 'hypercubes':
            Z_Q = F.sigmoid(Z_Q)
        elif noise_type == 'unitsphere':
            Z_Q = Z_Q / (torch.sqrt((Z_Q**2).sum(1, keepdim=True)) + 1e-6)
        elif noise_type == 'unitball':
            Z_Q = F.tanh(Z_Q)

    return Z_Q


def decode(models, Z, key='autoencoder'):
    decoder = models[key][1]

    return decoder(Z, nonlinearity=F.tanh)


# ROUTINES ================================================================================
# Each of these methods needs to take `data`, `models`, `losses`, `results`, and `viz`


def discriminator_routine(data,
                          models,
                          losses,
                          results,
                          viz,
                          measure='GAN',
                          noise_type=None,
                          output_nonlin=None,
                          noise='uniform'):
    X, Z_P, U = data.get_batch('images', 'y', 'u')
    Z_P = shape_noise(Z_P, U, noise_type)

    Z_Q = encode(models, X, output_nonlin=output_nonlin, noise_type=noise_type)
    E_pos, E_neg, _, _ = score(models, Z_P, Z_Q, measure)
    losses.update(discriminator=E_neg - E_pos)


def penalty_routine(data,
                    models,
                    losses,
                    results,
                    viz,
                    penalty_amount=0.5,
                    output_nonlin=None,
                    noise_type=None):
    X, Z_P, U = data.get_batch('images', 'y', 'u')
    Z_P = shape_noise(Z_P, U, noise_type)
    Z_P, Z_Q, Y_Q = encode(
        models, X, Z_P, output_nonlin=output_nonlin, noise_type=noise_type)

    penalty = apply_gradient_penalty(
        data,
        models,
        inputs=(Z_P, Z_Q),
        model='discriminator',
        penalty_amount=penalty_amount)

    if penalty:
        losses.discriminator = penalty


def main_routine(data,
                 models,
                 losses,
                 results,
                 viz,
                 measure=None,
                 noise_type='hypercubes',
                 output_nonlin=False,
                 generator_loss_type='non-saturating',
                 beta=1.0):
    X_P, Z_P, T, U = data.get_batch('images', 'y', 'targets', 'u')
    Z_P = shape_noise(Z_P, U, noise_type)

    Z_Q = encode(
        models, X_P, output_nonlin=output_nonlin, noise_type=noise_type)
    E_pos, E_neg, P_samples, Q_samples = score(models, Z_P, Z_Q, measure)
    get_results(P_samples, Q_samples, E_pos, E_neg, measure, results=results)
    visualize(Z_Q, P_samples, Q_samples, X_P, T, viz=viz)

    X_R = decode(models, Z_Q)
    X_G = decode(models, Z_P)
    reconstruction_loss = F.mse_loss(X_R, X_P) / X_P.size(0)
    encoder_loss = generator_loss(
        Q_samples, measure, loss_type=generator_loss_type)

    # correlations = cross_correlation(Z_Q, remove_diagonal=True)

    losses.autoencoder = beta * encoder_loss + reconstruction_loss
    results.update(
        reconstruction_loss=reconstruction_loss.item(),
        gan_loss=encoder_loss.item())
    # viz.add_heatmap(correlations.data, name='latent correlations')
    viz.add_image(X_G, name='generated')
    viz.add_image(X_P, name='ground truth')
    viz.add_image(X_R, name='reconstructed')


def classifier_routine(data,
                       models,
                       losses,
                       results,
                       viz,
                       criterion=torch.nn.CrossEntropyLoss()):
    X, Y = data.get_batch('images', 'targets')
    classifier = models.classifier
    Z = encode(models, X)

    classify(
        classifier, Z, Y, losses=losses, results=results, criterion=criterion)


# CORTEX ===============================================================================
# Must include `BUILD`, `TRAIN_ROUTINES`, and `DEFAULT_CONFIG`


def SETUP(model=None, data=None, routines=None, **kwargs):
    noise = routines.discriminator.noise
    noise_type = routines.discriminator.noise_type
    if noise_type in ('unitsphere', 'unitball'):
        noise = 'normal'
    data.noise_variables = dict(
        y=dict(dist=noise, size=model.dim_z), u=dict(dist='uniform', size=1))


def BUILD(data,
          models,
          encoder_type='convnet',
          decoder_type='convnet',
          dim_z=64,
          encoder_args={},
          decoder_args={}):
    x_shape = data.get_dims('x', 'y', 'c')
    dim_l = data.get_dims('labels')
    Encoder, encoder_args = update_encoder_args(
        x_shape, model_type=encoder_type, encoder_args=encoder_args)
    Decoder, decoder_args = update_decoder_args(
        x_shape, model_type=decoder_type, decoder_args=decoder_args)
    encoder = build_encoder(
        None,
        x_shape,
        dim_z,
        Encoder,
        fully_connected_layers=[1028],
        **encoder_args)
    decoder = build_decoder(None, x_shape, dim_z, Decoder, **decoder_args)
    classifier = FullyConnectedNet(
        dim_z, dim_h=[200, 200], dim_out=dim_l, batch_norm=True, dropout=0.2)
    build_discriminator(models, dim_z)

    models.update(autoencoder=(encoder, decoder), classifier=classifier)


TRAIN_ROUTINES = dict(
    discriminator=discriminator_routine,
    autoencoder=main_routine,
    classifier=classifier_routine)

INFO = dict(
    measure=dict(
        choices=['GAN', 'JSD', 'KL', 'RKL', 'X2', 'H2', 'DV', 'W1'],
        help='GAN measure. {GAN, JSD, KL, RKL (reverse KL), X2 (Chi^2),'
        ' H2 (squared Hellinger), DV (Donsker Varahdan KL), W1 (IPM)}'),
    noise_type=dict(
        choices=['hypercubes', 'unitball', 'unitsphere'],
        help='Type of noise to match encoder output to.'),
    noise=dict(help='Distribution of noise. (to be deprecacated).'),
    output_nonlin=dict(
        help='Apply nonlinearity at the output of encoder. Will be chosen'
        'according to `noise_type`.'),
    generator_loss_type=dict(
        choices=['non-saturating', 'minimax', 'boundary-seek'],
        help='Generator loss type.'),
    beta=dict(help='Beta scaling for GAN term in autoencoder.'),
    penalty_amount=dict(
        help='Amount of gradient penalty for the discriminator.'),
    model_type=dict(choices=['mnist', 'convnet', 'resnet'], help='Model type.'),
    dim_z=dict(help='Latent dimension.'))

DEFAULT_CONFIG = dict(
    data=dict(batch_size=dict(train=64, test=640)),
    optimizer=dict(optimizer='Adam', learning_rate=1e-4),
    train=dict(epochs=500, archive_every=10))
