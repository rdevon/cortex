'''MINE feature detection

'''

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from .ali import build_discriminator, build_extra_networks, score, apply_penalty
from .classifier import classify
from .featnet import (apply_gradient_penalty, build_encoder, build_discriminator as build_noise_discriminator, encode,
                      get_results, score as featnet_score, shape_noise, visualize)
from .gan import generator_loss
from .utils import cross_correlation
from .vae import update_decoder_args, update_encoder_args


resnet_encoder_args_ = dict(dim_h=64, batch_norm=True, f_size=3, n_steps=3)
resnet_decoder_args_ = dict(dim_h=64, batch_norm=True, f_size=3, n_steps=3)
mnist_encoder_args_ = dict(dim_h=64, batch_norm=True, f_size=5, pad=2, stride=2, min_dim=7)
mnist_decoder_args_ = dict(dim_h=64, batch_norm=True, f_size=4, pad=1, stride=2, n_steps=2)
convnet_encoder_args_ = dict(dim_h=64, batch_norm=True, n_steps=3)
convnet_decoder_args_ = dict(dim_h=64, batch_norm=True, n_steps=3)


def setup(model=None, data=None, routines=None, **kwargs):
    noise = routines['noise_discriminator']['noise']
    noise_type = routines['noise_discriminator']['noise_type']
    if noise_type in ('unitsphere', 'unitball'):
        noise = 'normal'
    data['noise_variables'] = dict(y=(noise, model['dim_noise']))
    data['noise_variables']['u'] = ('uniform', 1)
    routines['discriminator'].update(noise=noise, noise_type=noise_type)
    routines['encoder'].update(**routines['discriminator'])
    routines['encoder'].update(noise_measure=routines['noise_discriminator']['measure'])


def encoder_routine(data, models, losses, results, viz, measure=None, noise_measure=None, noise_type='hypercubes',
                    output_nonlin=False, generator_loss_type=None, **kwargs):
    X_P, X_Q, T, Y_P, U = data.get_batch('1.images', '2.images', '1.targets', 'y', 'u')

    Z_P, Z, Y_Q = encode(models, X_P, Y_P, output_nonlin=output_nonlin, noise_type=noise_type)
    E_pos, E_neg, P_samples, Q_samples = score(models, X_P, X_Q, Z, Z, measure)
    E_pos_KL, E_neg_KL, _, _ = score(models, X_P, X_Q, Z, Z, 'KL')
    E_pos_DV, E_neg_DV, _, _ = score(models, X_P, X_Q, Z, Z, 'DV')

    losses.update(encoder=E_neg - E_pos)
    results.update(Mutual_Information=dict(DV=E_pos_DV-E_neg_DV,KL=E_pos_KL-E_neg_KL))
    get_results(P_samples, Q_samples, E_pos, E_neg, measure, results=results)
    visualize(Z, P_samples, Q_samples, X_P, T, Y_Q=Y_Q, viz=viz)

    if 'noise_discriminator' in models:
        Y_P = shape_noise(Y_P, U, noise_type)
        E_pos_n, E_neg_n, P_samples_n, Q_samples_n = featnet_score(models, Z_P, Z, measure, Y_P=Y_P, Y_Q=Y_Q,
                                                             key='noise_discriminator')
        results_ = {}
        get_results(P_samples_n, Q_samples_n, E_pos_n, E_neg_n, noise_measure, results={})
        results_ = dict(('noise_' + k, v) for k, v in results_.items())
        results.update(**results_)
        losses['encoder'] += generator_loss(Q_samples_n, noise_measure, loss_type=generator_loss_type)


def discriminator_routine(data, models, losses, results, viz, measure=None, penalty_amount=None, output_nonlin=False,
                          noise_type='hypercubes', **kwargs):
    X_P, X_Q, T = data.get_batch('1.images', '2.images', '1.targets')

    _, Z, Y_Q = encode(models, X_P, None, output_nonlin=output_nonlin, noise_type=noise_type)
    E_pos, E_neg, _, _ = score(models, X_P, X_Q, Z, Z, measure)

    losses.update(discriminator=E_neg - E_pos)
    apply_penalty(models, losses, results, X_P, Z, penalty_amount)


def noise_discriminator_routine(data, models, losses, results, viz, penalty_amount=0., measure=None,
                                noise_type='hypercubes', output_nonlin=False, **kwargs):
    X, Y_P, U = data.get_batch('1.images', 'y', 'u')
    Y_P = shape_noise(Y_P, U, noise_type)

    Z_P, Z_Q, Y_Q = encode(models, X, Y_P, output_nonlin=output_nonlin, noise_type=noise_type)
    E_pos, E_neg, _, _ = featnet_score(models, Z_P, Z_Q, measure, Y_P=Y_P, Y_Q=Y_Q, key='noise_discriminator')
    losses.update(noise_discriminator=E_neg - E_pos)

    if Y_Q is not None:
        Z_Q = torch.cat([Y_Q, Z_Q], 1)
        Z_P = torch.cat([Y_P, Z_P], 1)

    apply_gradient_penalty(data, models, losses, results, inputs=(Z_P, Z_Q), model='noise_discriminator',
                           penalty_amount=penalty_amount)


def network_routine(data, models, losses, results, viz):
    X, Y = data.get_batch('1.images', '1.targets')
    encoder = models['encoder']
    if isinstance(encoder, (list, tuple)):
        encoder = encoder[0]
    classifier, decoder = models['nets']
    Z_P = encoder(X)

    Z_t = Variable(Z_P.data.cuda(), requires_grad=False)
    X_d = decoder(Z_t, nonlinearity=F.tanh)
    dd_loss = ((X - X_d) ** 2).sum(1).sum(1).sum(1).mean()
    classify(classifier, Z_P, Y, losses=losses, results=results, key='nets')
    losses['nets'] += dd_loss

    correlations = cross_correlation(Z_P, remove_diagonal=True)
    viz.add_heatmap(correlations.data, name='latent correlations')
    viz.add_image(X_d, name='Reconstruction')


def build_model(data, models, model_type='convnet', dim_embedding=None, dim_noise=None,
                encoder_args=None, decoder_args=None, use_topnet=None, match_noise=None):

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
    build_encoder(models, x_shape, dim_noise, Encoder, use_topnet=use_topnet, dim_top=dim_noise, **encoder_args)
    build_extra_networks(models, x_shape, dim_embedding, dim_l, Decoder, **decoder_args)

    if match_noise:
        build_noise_discriminator(models, dim_d, key='noise_discriminator')


ROUTINES = dict(discriminator=discriminator_routine, noise_discriminator=noise_discriminator_routine,
                encoder=encoder_routine, nets=network_routine)


DEFAULT_CONFIG = dict(
    data=dict(batch_size=dict(train=64, test=1028), skip_last_batch=True, duplicate=2),
    optimizer=dict( optimizer='Adam', learning_rate=1e-4),
    model=dict(model_type='convnet', dim_embedding=64, dim_noise=64, match_noise=False, use_topnet=False,
               encoder_args=None),
    routines=dict(discriminator=dict(measure='JSD', penalty_amount=1.),
                  noise_discriminator=dict(measure='JSD', penalty_amount=1., noise_type='hypercubes', noise='uniform'),
                  encoder=dict(generator_loss_type='non-saturating'),
                  nets=dict()),
    train=dict(epochs=2000, archive_every=10)
)
