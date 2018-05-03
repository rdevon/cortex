'''Adversarially learned inference and Bi-GAN
'''

import torch
import torch.nn.functional as F
from torch import autograd

from .classifier import classify
from .gan import get_positive_expectation, get_negative_expectation
from .modules.fully_connected import FullyConnectedNet
from .utils import cross_correlation
from .vae import update_decoder_args, update_encoder_args, build_encoder, build_decoder


def setup(model=None, data=None, routines=None, **kwargs):
    data['noise_variables']['z']['size'] = model['dim_z']
    routines['generator']['measure'] = routines['discriminator']['measure']


def apply_penalty(models, losses, results, X, Z, penalty_amount, key='discriminator'):
    x_disc, z_disc, topnet = models[key]
    if penalty_amount:
        X = X.detach()
        Z = Z.detach()
        X.requires_grad_()
        Z.requires_grad_()
        W = x_disc(X, nonlinearity=F.relu)
        U = z_disc(Z, nonlinearity=F.relu)
        S = topnet(torch.cat([W, U], 1))

        G = autograd.grad(outputs=S, inputs=[X, Z], grad_outputs=torch.ones(S.size()).cuda(),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]

        G = G.view(G.size()[0], -1)
        G = (G ** 2).sum(1).mean()
        losses[key] += penalty_amount * G
        results['gradient penalty'] = G.item()


def score(models, X_P, X_Q, Z_P, Z_Q, measure, key='discriminator'):
    x_disc, z_disc, topnet = models[key]
    W_Q = x_disc(X_Q, nonlinearity=F.relu)
    W_P = x_disc(X_P, nonlinearity=F.relu)
    U_Q = z_disc(Z_Q, nonlinearity=F.relu)
    U_P = z_disc(Z_P, nonlinearity=F.relu)

    P_samples = topnet(torch.cat([W_P, U_P], 1))
    Q_samples = topnet(torch.cat([W_Q, U_Q], 1))

    E_pos = get_positive_expectation(P_samples, measure)
    E_neg = get_negative_expectation(Q_samples, measure)

    return E_pos, E_neg, P_samples, Q_samples

def discriminator_routine(data, models, losses, results, viz, measure=None, penalty_amount=None):
    X_P, T, Z_Q = data.get_batch('images', 'targets', 'z')
    encoder, decoder = models['generator']

    X_Q = decoder(Z_Q, nonlinearity=F.tanh)
    Z_P = encoder(X_P)

    E_pos, E_neg, P_samples, Q_samples = score(models, X_P, X_Q, Z_P, Z_Q, measure)
    difference = E_pos - E_neg

    losses.update(discriminator=-difference)
    apply_penalty(models, losses, results, X_P, Z_P, penalty_amount)

    results.update(Scores=dict(Ep=P_samples.mean().item(), Eq=Q_samples.mean().item()))
    results['{} distance'.format(measure)] = difference.item()
    viz.add_image(X_Q, name='generated')
    viz.add_histogram(dict(fake=Q_samples.view(-1).data, real=P_samples.view(-1).data), name='discriminator output')
    viz.add_scatter(Z_P, labels=T.data, name='latent values')


def generator_routine(data, models, losses, results, viz, measure=None):
    X_P, Z_Q = data.get_batch('images', 'z')
    encoder, decoder = models['generator']

    X_Q = decoder(Z_Q, nonlinearity=F.tanh)
    Z_P = encoder(X_P)

    E_pos, E_neg, P_samples, Q_samples = score(models, X_P, X_Q, Z_P, Z_Q, measure)
    difference = E_pos - E_neg

    losses.update(generator=difference)


def network_routine(data, models, losses, results, viz, encoder_key='generator'):
    X, Y = data.get_batch('images', 'targets')
    encoder = models[encoder_key]
    if isinstance(encoder, (list, tuple)):
        encoder = encoder[0]
    classifier, decoder = models['nets']
    Z_P = encoder(X)

    Z_t = Z_P.detach()
    X_d = decoder(Z_t, nonlinearity=F.tanh)
    dd_loss = ((X - X_d) ** 2).sum(1).sum(1).sum(1).mean()
    classify(classifier, Z_P, Y, losses=losses, results=results, key='nets')
    losses['nets'] += dd_loss

    correlations = cross_correlation(Z_P, remove_diagonal=True)
    viz.add_heatmap(correlations.data, name='latent correlations')
    viz.add_image(X, name='Ground truth')
    viz.add_image(X_d, name='Reconstruction')


def build_discriminator(models, x_shape, dim_z, Encoder, key='discriminator', **encoder_args):
    discriminator_args = {}
    discriminator_args.update(**encoder_args)
    discriminator_args.update(fully_connected_layers=[], batch_norm=False)
    logger.debug('Forming discriminator with class {} and args: {}'.format(Encoder, discriminator_args))

    x_disc = Encoder(x_shape, dim_out=256, **discriminator_args)
    z_disc = FullyConnectedNet(dim_z, dim_h=[dim_z], dim_out=256)
    topnet = FullyConnectedNet(2 * 256, dim_h=[512, 128], dim_out=1, batch_norm=False)
    models[key]= (x_disc, z_disc, topnet)


def build_extra_networks(models, x_shape, dim_z, dim_l, Decoder, dropout=0.1,
                         **decoder_args):
    logger.debug('Forming dencoder with class {} and args: {}'.format(Decoder, decoder_args))
    decoder = Decoder(x_shape, dim_in=dim_z, **decoder_args)
    classifier = FullyConnectedNet(dim_z, dim_h=[200, 200], dim_out=dim_l, dropout=dropout, batch_norm=True)
    models.update(nets=(classifier, decoder))


def build_model(data, models, model_type='convnet', dim_z=64, encoder_args=None, decoder_args=None):
    x_shape = data.get_dims('x', 'y', 'c')
    dim_l = data.get_dims('labels')
    Encoder, encoder_args = update_encoder_args(x_shape, model_type=model_type, encoder_args=encoder_args)
    Decoder, decoder_args = update_decoder_args(x_shape, model_type=model_type, decoder_args=decoder_args)
    encoder = build_encoder(None, x_shape, dim_z, Encoder, fully_connected_layers=[1028], **encoder_args)
    decoder = build_decoder(None, x_shape, dim_z, Decoder, **decoder_args)
    models.update(generator=(encoder, decoder))
    build_discriminator(models, x_shape, dim_z, Encoder, **encoder_args)
    build_extra_networks(models, x_shape, dim_z, dim_l, Decoder, **decoder_args)


ROUTINES = dict(discriminator=discriminator_routine, generator=generator_routine, nets=network_routine)


DEFAULT_CONFIG = dict(
    data=dict(batch_size=dict(train=64, test=640),
              noise_variables=dict(z=dict(dist='normal', size=64))),
    optimizer=dict(
        optimizer='Adam',
        learning_rate=1e-4,
        updates_per_model=dict(discriminator=1, generator=1, nets=1)
    ),
    model=dict(model_type='convnet', dim_z=64, encoder_args=None, decoder_args=None),
    routines=dict(discriminator=dict(measure='GAN', penalty_amount=1.0),
                  generator=dict(),
                  nets=dict()),
    train=dict(
        epochs=500,
        archive_every=10
    )
)