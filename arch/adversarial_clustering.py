'''Adversarial clustering.

'''

from sklearn import metrics
import torch

from ali import build_discriminator as build_mine_discriminator, build_extra_networks, score, apply_penalty
from featnet import (apply_gradient_penalty, build_encoder, build_discriminator as build_noise_discriminator, encode,
                     get_results, score as featnet_score, visualize)
from minet import network_routine
from gan import generator_loss
from utils import update_decoder_args, update_encoder_args


def encoder_routine(data, models, losses, results, viz, mine_measure=None, noise_measure=None, noise_type='hypercubes',
                    output_nonlin=False, generator_loss_type=None, **kwargs):
    X_P, X_Q, T, Y_P = data.get_batch('1.images', '2.images', '1.targets', 'y')

    Z_P, Z, Y_Q = encode(models, X_P, Y_P, output_nonlin=output_nonlin, noise_type=noise_type)
    C = Z.argmax(1).data.cpu().numpy()
    ARI = metrics.adjusted_rand_score(T, C)
    AMI = metrics.adjusted_mutual_info_score(T, C)
    homogeneity = metrics.homogeneity_score(T, C)
    completeness = metrics.completeness_score(T, C)
    v_measure = metrics.v_measure_score(T, C)
    FMI = metrics.fowlkes_mallows_score(T, C)
    results.update(Cluster_scores=dict(ARI=ARI, AMI=AMI, homogeneity=homogeneity, completeness=completeness,
                                       v_measure=v_measure, FMI=FMI))
    E_pos, E_neg, P_samples, Q_samples = score(models, X_P, X_Q, Z, Z, mine_measure, key='mine_discriminator')

    losses.encoder = E_neg - E_pos
    get_results(P_samples, Q_samples, E_pos, E_neg, mine_measure, results=results, name='mine')
    visualize(Z, P_samples, Q_samples, X_P, T, Y_Q=Y_Q, viz=viz)

    E_pos_n, E_neg_n, P_samples_n, Q_samples_n = featnet_score(models, Z_P, Z, noise_measure, Y_P=Y_P, Y_Q=Y_Q,
                                                               key='noise_discriminator')
    get_results(P_samples_n, Q_samples_n, E_pos_n, E_neg_n, noise_measure, results=results, name='noise')
    losses.encoder += generator_loss(Q_samples_n, noise_measure, loss_type=generator_loss_type)


def mine_discriminator_routine(data, models, losses, results, viz, measure=None, penalty_amount=None, output_nonlin=False,
                               noise_type='hypercubes', **kwargs):
    X_P, X_Q, T = data.get_batch('1.images', '2.images', '1.targets')

    _, Z, Y_Q = encode(models, X_P, None, output_nonlin=output_nonlin, noise_type=noise_type)
    E_pos, E_neg, _, _ = score(models, X_P, X_Q, Z, Z, measure, key='mine_discriminator')

    losses.mine_discriminator = E_neg - E_pos
    penalty = apply_penalty(models, losses, results, X_P, Z, penalty_amount, key='mine_discriminator')

    # somethign here

def noise_discriminator_routine(data, models, losses, results, viz, penalty_amount=0., measure=None,
                                noise_type='hypercubes', output_nonlin=False, **kwargs):
    X, Y_P = data.get_batch('1.images', 'y')

    Z_P, Z_Q, Y_Q = encode(models, X, Y_P, output_nonlin=output_nonlin, noise_type=noise_type)
    E_pos, E_neg, _, _ = featnet_score(models, Z_P, Z_Q, measure, Y_P=Y_P, Y_Q=Y_Q, key='noise_discriminator')
    losses.noise_discriminator = E_neg - E_pos

    if Y_Q is not None:
        Z_Q = torch.cat([Y_Q, Z_Q], 1)
        Z_P = torch.cat([Y_P, Z_P], 1)

    penalty = apply_gradient_penalty(data, models, inputs=(Z_P, Z_Q), model='noise_discriminator', penalty_amount=penalty_amount)
    if penalty:
        losses.noise_discriminator += penalty


def SETUP(model=None, data=None, routines=None, **kwargs):
    noise_type = 'dirichlet'
    data.noise_variables = dict(y=dict(dist=noise_type, size=model.dim_noise, concentration=0.2))
    routines.encoder.mine_measure = routines.mine_discriminator.measure
    routines.encoder.noise_measure = routines.noise_discriminator.measure
    routines.mine_discriminator.output_nonlin = routines.encoder.output_nonlin
    routines.noise_discriminator.output_nonlin = routines.encoder.output_nonlin


def BUILD(data, models, encoder_type='convnet', decoder_type='convnet', dim_embedding=None, dim_noise=None,
          encoder_args=None, decoder_args=None, use_topnet=None):

    if not use_topnet:
        dim_embedding = dim_noise
        dim_d = dim_embedding
    else:
        dim_d = dim_embedding + dim_noise

    x_shape = data.get_dims('x', 'y', 'c')
    dim_l = data.get_dims('labels')

    Encoder, encoder_args = update_encoder_args(x_shape, model_type=encoder_type, encoder_args=encoder_args)
    Decoder, decoder_args = update_decoder_args(x_shape, model_type=decoder_type, decoder_args=decoder_args)
    build_mine_discriminator(models, x_shape, dim_embedding, Encoder, key='mine_discriminator', **encoder_args)
    build_noise_discriminator(models, dim_d, key='noise_discriminator')
    build_encoder(models, x_shape, dim_noise, Encoder, use_topnet=use_topnet, dim_top=dim_noise, **encoder_args)
    build_extra_networks(models, x_shape, dim_embedding, dim_l, Decoder, **decoder_args)


TRAIN_ROUTINES = dict(mine_discriminator=mine_discriminator_routine, noise_discriminator=noise_discriminator_routine,
                      encoder=encoder_routine, nets=network_routine)


DEFAULT_CONFIG = dict(
    data=dict(batch_size=dict(train=64, test=1028), duplicate=2),
    optimizer=dict( optimizer='Adam', learning_rate=1e-4,
                    updates_per_routine=dict(noise_discriminator=1, mine_discriminator=1, autoencoder=1, classifier=1)),
    model=dict(dim_embedding=64, dim_noise=64, use_topnet=False, encoder_args=None),
    routines=dict(mine_discriminator=dict(measure='JSD', penalty_amount=0.5),
                  noise_discriminator=dict(measure='JSD', penalty_amount=0.2),
                  encoder=dict(generator_loss_type='non-saturating', output_nonlin=torch.nn.Softmax(dim=1)),
                  nets=dict()),
    train=dict(epochs=2000, archive_every=10))