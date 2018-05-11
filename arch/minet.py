'''MINE feature detection

'''

import torch
import torch.nn.functional as F

from ali import build_discriminator, build_extra_networks, score, apply_penalty
from classifier import classify
from featnet import (apply_gradient_penalty, build_encoder, build_discriminator as build_noise_discriminator, encode,
                     get_results, score as featnet_score, shape_noise, visualize)
from gan import generator_loss
from utils import cross_correlation, ms_ssim
from vae import update_decoder_args, update_encoder_args


# ROUTINES =============================================================================================================
# Each of these methods needs to take `data`, `models`, `losses`, `results`, and `viz`

def encoder_routine(data, models, losses, results, viz, measure=None, noise_measure=None, noise_type=None,
                    output_nonlin=False, generator_loss_type='non-saturating', beta=1.0, key='discriminator'):
    X_P, X_Q, T, Y_P, U = data.get_batch('1.images', '2.images', '1.targets', 'y', 'u')

    Z_P, Z, Y_Q = encode(models, X_P, Y_P, output_nonlin=output_nonlin, noise_type=noise_type)
    E_pos, E_neg, P_samples, Q_samples = score(models, X_P, X_Q, Z, Z, measure, key=key)

    losses.encoder = E_neg - E_pos
    get_results(P_samples, Q_samples, E_pos, E_neg, measure, results=results, name='mine')
    visualize(Z, P_samples, Q_samples, X_P, T, Y_Q=Y_Q, viz=viz)

    if 'noise_discriminator' in models:
        Y_P = shape_noise(Y_P, U, noise_type)
        E_pos_n, E_neg_n, P_samples_n, Q_samples_n = featnet_score(models, Z_P, Z, measure, Y_P=Y_P, Y_Q=Y_Q,
                                                                   key='noise_discriminator')
        get_results(P_samples_n, Q_samples_n, E_pos_n, E_neg_n, noise_measure, results=results, name='noise')
        losses.encoder += beta * generator_loss(Q_samples_n, noise_measure, loss_type=generator_loss_type)


def discriminator_routine(data, models, losses, results, viz, measure='JSD', penalty_amount=0.5, output_nonlin=None,
                          noise_type='hypercubes', noise='uniform'):
    X_P, X_Q, T = data.get_batch('1.images', '2.images', '1.targets')

    _, Z, Y_Q = encode(models, X_P, None, output_nonlin=output_nonlin, noise_type=noise_type)
    E_pos, E_neg, _, _ = score(models, X_P, X_Q, Z, Z, measure)

    losses.discriminator = E_neg - E_pos


def noise_discriminator_routine(data, models, losses, results, viz, noise_penalty_amount=0.5, noise_measure='JSD',
                                noise_type=None, output_nonlin=None):
    X, Y_P, U = data.get_batch('1.images', 'y', 'u')
    Y_P = shape_noise(Y_P, U, noise_type)

    Z_P, Z_Q, Y_Q = encode(models, X, Y_P, output_nonlin=output_nonlin, noise_type=noise_type)
    E_pos, E_neg, _, _ = featnet_score(models, Z_P, Z_Q, noise_measure, Y_P=Y_P, Y_Q=Y_Q, key='noise_discriminator')

    if Y_Q is not None:
        Z_Q = torch.cat([Y_Q, Z_Q], 1)
        Z_P = torch.cat([Y_P, Z_P], 1)

    penalty = apply_gradient_penalty(data, models, inputs=(Z_P, Z_Q), model='noise_discriminator',
                                     penalty_amount=noise_penalty_amount)

    losses.noise_discriminator = E_neg - E_pos + penalty


def network_routine(data, models, losses, results, viz):
    X, Y = data.get_batch('1.images', '1.targets')
    encoder = models.encoder
    if isinstance(encoder, (list, tuple)):
        encoder = encoder[0]
    classifier, decoder = models.nets

    Z_P = encoder(X)
    Z_t = Z_P.detach()
    X_d = decoder(Z_t, nonlinearity=F.tanh)
    dd_loss = ((X - X_d) ** 2).sum(1).sum(1).sum(1).mean()
    classify(classifier, Z_P, Y, losses=losses, results=results, key='nets')

    #correlations = cross_correlation(Z_P, remove_diagonal=True)
    msssim = ms_ssim(X, X_d).item()

    losses.nets += dd_loss
    results.update(reconstruction_loss=dd_loss.item(), ms_ssim=msssim)
    #viz.add_heatmap(correlations.data, name='latent correlations')
    viz.add_image(X_d, name='Reconstruction')

# CORTEX ===============================================================================================================
# Must include `BUILD`, `TRAIN_ROUTINES`, and `DEFAULT_CONFIG`

def SETUP(model=None, data=None, routines=None, **kwargs):
    noise = routines.discriminator.noise
    noise_type = routines.discriminator.noise_type
    if noise_type in ('unitsphere', 'unitball'):
        noise = 'normal'
    data.noise_variables = dict(y=dict(dist=noise, size=model.dim_noise),
                                u=dict(dist='uniform', size=1))


def BUILD(data, models, model_type='convnet', dim_embedding=64, dim_noise=64,
          encoder_args={}, decoder_args={}, use_topnet=False, match_noise=False, add_supervision=False):
    global TRAIN_ROUTINES

    if not use_topnet:
        dim_embedding = dim_noise
        dim_d = dim_embedding
    else:
        dim_d = dim_embedding + dim_noise

    x_shape = data.get_dims('x', 'y', 'c')
    dim_l = data.get_dims('labels')

    Encoder, encoder_args = update_encoder_args(x_shape, model_type=model_type, encoder_args=encoder_args)
    Decoder, decoder_args = update_decoder_args(x_shape, model_type=model_type, decoder_args=decoder_args)
    build_discriminator(models, x_shape, dim_embedding, Encoder, **encoder_args)
    build_encoder(models, x_shape, dim_noise, Encoder, fully_connected_layers=[1028], use_topnet=use_topnet,
                  dim_top=dim_noise, **encoder_args)

    if match_noise:
        build_noise_discriminator(models, dim_d, key='noise_discriminator')
    else:
        TRAIN_ROUTINES.pop('noise_discriminator') # HACKY

    if add_supervision:
        build_extra_networks(models, x_shape, dim_embedding, dim_l, Decoder, **decoder_args)
        TRAIN_ROUTINES.update(nets=network_routine)


TRAIN_ROUTINES = dict(discriminator=discriminator_routine, encoder=encoder_routine,
                      noise_discriminator=noise_discriminator_routine)

INFO = dict(measure=dict(choices=['GAN', 'JSD', 'KL', 'RKL', 'X2', 'H2', 'DV', 'W1'],
                         help='GAN measure. {GAN, JSD, KL, RKL (reverse KL), X2 (Chi^2), H2 (squared Hellinger), '
                              'DV (Donsker Varahdan KL), W1 (IPM)}'),
            noise_type=dict(choices=['hypercubes', 'unitball', 'unitsphere'],
                            help='Type of noise to match encoder output to.'),
            noise=dict(help='Distribution of noise. (to be deprecacated).'),
            output_nonlin=dict(help='Apply nonlinearity at the output of encoder. Will be chosen according to `noise_type`.'),
            generator_loss_type=dict(choices=['non-saturating', 'minimax', 'boundary-seek'],
                                     help='Generator loss type.'),
            penalty_amount=dict(help='Amount of gradient penalty for the discriminator.'),
            model_type=dict(choices=['mnist', 'convnet', 'resnet'],
                            help='Model type.'),
            dim_noise=dict(help='Noise dimension.'),
            dim_embedding=dict(help='Embedding dimension (used if use_topnet is True)'),
            use_topnet=dict(help='Whether to use an additional network and perform ALI with top two variables.'),
            match_noise=dict(help='Use another discriminator to match the output to noise.'),
            add_supervision=dict(help='Use additional networks for monitoring during training.')
)

DEFAULT_CONFIG = dict(
    data=dict(batch_size=dict(train=64, test=640), duplicate=2),
    train=dict(epochs=2000, archive_every=10, save_on_lowest='losses.encoder')
)
