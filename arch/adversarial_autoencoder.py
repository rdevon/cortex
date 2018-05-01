'''Adversarial autoencoder
'''

import torch
import torch.nn.functional as F

from .classifier import classify
from .gan import generator_loss
from .featnet import apply_gradient_penalty, build_discriminator, get_results, score, shape_noise, visualize
from .modules.fully_connected import FullyConnectedNet
from .utils import cross_correlation
from .vae import update_decoder_args, update_encoder_args, build_encoder, build_decoder


def encode(models, X, output_nonlin=False, noise_type='hypercubes', key='autoencoder'):
    encoder = models[key][0]

    Z_Q = encoder(X)
    if output_nonlin:
        if noise_type == 'hypercubes':
            Z_Q = F.sigmoid(Z_Q)
        elif noise_type == 'unitsphere':
            Z_Q = Z_Q / (torch.sqrt((Z_Q ** 2).sum(1, keepdim=True)) + 1e-6)
        elif noise_type == 'unitball':
            Z_Q = F.tanh(Z_Q)

    return Z_Q


def decode(models, Z, key='autoencoder'):
    decoder = models[key][1]

    return decoder(Z, nonlinearity=F.tanh)


def discriminator_routine(data, models, losses, results, viz, penalty_amount=0., measure=None, noise_type='hypercubes',
                          output_nonlin=False, **kwargs):
    X, Z_P, U = data.get_batch('images', 'y', 'u')
    Z_P = shape_noise(Z_P, U, noise_type)

    Z_Q = encode(models, X, output_nonlin=output_nonlin, noise_type=noise_type)
    E_pos, E_neg, _, _ = score(models, Z_P, Z_Q, measure)
    losses.update(discriminator=E_neg - E_pos)

    apply_gradient_penalty(data, models, losses, results, inputs=(Z_P, Z_Q), model='discriminator',
                           penalty_amount=penalty_amount)


def main_routine(data, models, losses, results, viz, measure=None, noise_type='hypercubes',
                 output_nonlin=False, generator_loss_type=None, **kwargs):
    X_P, Z_P, T, U = data.get_batch('images', 'y', 'targets', 'u')
    Z_P = shape_noise(Z_P, U, noise_type)

    Z_Q = encode(models, X_P, output_nonlin=output_nonlin, noise_type=noise_type)
    E_pos, E_neg, P_samples, Q_samples = score(models, Z_P, Z_Q, measure)
    get_results(P_samples, Q_samples, E_pos, E_neg, measure, results=results)
    visualize(Z_Q, P_samples, Q_samples, X_P, T, viz=viz)

    X_R = decode(models, Z_Q)
    X_G = decode(models, Z_P)
    reconstruction_loss = F.mse_loss(X_R, X_P, size_average=False)
    encoder_loss = generator_loss(Q_samples, measure, loss_type=generator_loss_type)
    losses.update(autoencoder=encoder_loss + reconstruction_loss)
    results.update(reconstruction_loss=reconstruction_loss.item(), gan_loss=encoder_loss.item())

    correlations = cross_correlation(Z_Q, remove_diagonal=True)
    
    viz.add_heatmap(correlations.data, name='latent correlations')
    viz.add_image(X_G, name='generated')
    viz.add_image(X_P, name='ground truth')
    viz.add_image(X_R, name='reconstructed')


def classifier_routine(data, models, losses, results, viz, **kwargs):
    X, Y = data.get_batch('images', 'targets')
    classifier = models['classifier']
    Z = encode(models, X)

    classify(classifier, Z, Y, losses=losses, results=results, **kwargs)


def setup(model=None, data=None, routines=None, **kwargs):
    noise = routines['discriminator']['noise']
    noise_type = routines['discriminator']['noise_type']
    if noise_type in ('unitsphere', 'unitball'):
        noise = 'normal'
    data['noise_variables'] = dict(y=dict(dist=noise, size=model['dim_z']))
    data['noise_variables']['u'] = dict(dist='uniform', size=1)
    routines['autoencoder'].update(**routines['discriminator'])


def build_model(data, models, model_type='convnet', dim_z=None, encoder_args=None, decoder_args=None):
    x_shape = data.get_dims('x', 'y', 'c')
    dim_l = data.get_dims('labels')
    Encoder, encoder_args = update_encoder_args(x_shape, model_type=model_type, encoder_args=encoder_args)
    Decoder, decoder_args = update_decoder_args(x_shape, model_type=model_type, decoder_args=decoder_args)
    encoder = build_encoder(None, x_shape, dim_z, Encoder, fully_connected_layers=[1028], **encoder_args)
    decoder = build_decoder(None, x_shape, dim_z, Decoder, **decoder_args)
    classifier = FullyConnectedNet(dim_z, dim_h=[200, 200], dim_out=dim_l, batch_norm=True, dropout=0.2)
    build_discriminator(models, dim_z)

    models.update(autoencoder=(encoder, decoder), classifier=classifier)


ROUTINES = dict(discriminator=discriminator_routine, autoencoder=main_routine, classifier=classifier_routine)


DEFAULT_CONFIG = dict(
    data=dict(batch_size=dict(train=64, test=640)),
    optimizer=dict(optimizer='Adam', learning_rate=1e-4),
    model=dict(model_type='convnet', dim_z=64, encoder_args=None, decoder_args=None),
    routines=dict(discriminator=dict(measure='GAN', penalty_amount=0.5, noise_type='hypercubes', noise='uniform'),
                  autoencoder=dict(generator_loss_type='non-saturating'),
                  classifier=dict()),
    train=dict(
        epochs=500,
        archive_every=10
    )
)