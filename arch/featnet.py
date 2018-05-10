'''Implicit feature network

'''

import numpy as np
import torch
import torch.nn.functional as F

from .ali import build_extra_networks, network_routine as ali_network_routine
from .gan import get_positive_expectation, get_negative_expectation, apply_gradient_penalty, generator_loss
from .modules.fully_connected import FullyConnectedNet
from .vae import update_encoder_args, update_decoder_args
from .utils import perform_svc


# Helper functions =====================================================================================================

def shape_noise(Y_P, U, noise_type, epsilon=1e-6):
    if noise_type == 'hypercubes':
        pass
    elif noise_type == 'unitsphere':
        Y_P = Y_P / (torch.sqrt((Y_P ** 2).sum(1, keepdim=True)) + epsilon)
    elif noise_type == 'unitball':
        Y_P = Y_P / (torch.sqrt((Y_P ** 2).sum(1, keepdim=True)) + epsilon) * U.expand(Y_P.size())
    else:
        raise ValueError

    return Y_P


def encode(models, X, Y_P, output_nonlin=False, noise_type='hypercubes', key='encoder'):
    encoder = models[key]

    if isinstance(encoder, (tuple, list)) and len(encoder) == 3:
        encoder, topnet, revnet = encoder
    else:
        topnet = None

    Z_Q = encoder(X)

    if callable(output_nonlin):
        Z_Q = output_nonlin(Z_Q)
    elif output_nonlin:
        if noise_type == 'hypercubes':
            Z_Q = F.sigmoid(Z_Q)
        elif noise_type == 'unitsphere':
            Z_Q = Z_Q / (torch.sqrt((Z_Q ** 2).sum(1, keepdim=True)) + 1e-6)
        elif noise_type == 'unitball':
            Z_Q = F.tanh(Z_Q)

    if topnet is not None:
        Y_Q = topnet(Z_Q)
        Z_P = revnet(Y_P)
    else:
        Y_Q = None
        Z_P = Y_P

    return Z_P, Z_Q, Y_Q


def score(models, Z_P, Z_Q, measure, Y_P=None, Y_Q=None, key='discriminator'):
    discriminator = models[key]
    if Y_Q is not None:
        Z_P = torch.cat([Y_P, Z_P], 1)
        Z_Q = torch.cat([Y_Q, Z_Q], 1)

    P_samples = discriminator(Z_P)
    Q_samples = discriminator(Z_Q)

    E_pos = get_positive_expectation(P_samples, measure)
    E_neg = get_negative_expectation(Q_samples, measure)
    return E_pos, E_neg, P_samples, Q_samples


def get_results(P_samples, Q_samples, E_pos, E_neg, measure, results=None, name=None):
    if results is not None:
        score_name = 'Scores' if name is None else '{} Score'.format(name)
        distance_name = '{} distance'.format(measure) if name is None else '{} {} distance'.format(name, measure)
        results[score_name] = dict(Ep=P_samples.mean().item(), Eq=Q_samples.mean().item())
        results[distance_name] = (E_pos - E_neg).item()


def visualize(Z_Q, P_samples, Q_samples, X, T, Y_Q=None, viz=None):
    if viz is not None:
        if Y_Q is not None:
            viz.add_scatter(Z_Q, labels=T.data, name='intermediate values')
            viz.add_scatter(Y_Q, labels=T.data, name='latent values')
        else:
            viz.add_scatter(Z_Q, labels=T.data, name='latent values')
        viz.add_histogram(dict(real=P_samples.view(-1).data, fake=Q_samples.view(-1).data), name='discriminator output')

# ROUTINES =============================================================================================================
# Each of these methods needs to take `data`, `models`, `losses`, `results`, and `viz`

def encoder_routine(data, models, losses, results, viz, measure=None, noise_type='hypercubes',
                    output_nonlin=False, generator_loss_type=None):
    X, Y_P, T, U = data.get_batch('images', 'y', 'targets', 'u')
    Y_P = shape_noise(Y_P, U, noise_type)

    Z_P, Z_Q, Y_Q = encode(models, X, Y_P, output_nonlin=output_nonlin, noise_type=noise_type)
    E_pos, E_neg, P_samples, Q_samples = score(models, Z_P, Z_Q, measure, Y_P=Y_P, Y_Q=Y_Q)
    get_results(P_samples, Q_samples, E_pos, E_neg, measure, results=results)
    visualize(Z_Q, P_samples, Q_samples, X, T, Y_Q=Y_Q, viz=viz)

    encoder_loss = generator_loss(Q_samples, measure, loss_type=generator_loss_type)
    losses.encoder = encoder_loss


def discriminator_routine(data, models, losses, results, viz, measure=None, noise_type='hypercubes',
                          output_nonlin=False, noise=None):
    X, Y_P, U = data.get_batch('images', 'y', 'u')
    Y_P = shape_noise(Y_P, U, noise_type)

    Z_P, Z_Q, Y_Q = encode(models, X, Y_P, output_nonlin=output_nonlin, noise_type=noise_type)
    E_pos, E_neg, _, _ = score(models, Z_P, Z_Q, measure, Y_P=Y_P, Y_Q=Y_Q)
    losses.discriminator = E_neg - E_pos


def penalty_routine(data, models, losses, results, viz, penalty_amount=None, output_nonlin=False, noise_type=None):
    X, Y_P, U = data.get_batch('images', 'y', 'u')
    Y_P = shape_noise(Y_P, U, noise_type)
    Z_P, Z_Q, Y_Q = encode(models, X, Y_P, output_nonlin=output_nonlin, noise_type=noise_type)

    if Y_Q is not None:
        Z_Q = torch.cat([Y_Q, Z_Q], 1)
        Z_P = torch.cat([Y_P, Z_P], 1)

    penalty = apply_gradient_penalty(data, models, inputs=(Z_P, Z_Q), model='discriminator',
                                     penalty_amount=penalty_amount)

    if penalty:
        losses.discriminator = penalty


def network_routine(data, models, losses, results, viz, **kwargs):
    ali_network_routine(data, models, losses, results, viz, encoder_key='encoder', **kwargs)

# SVM routines =========================================================================================================

SVM = None

def collect_embeddings(data, models, encoder_key='encoder', test=False):
    encoder = models[encoder_key]
    if isinstance(encoder, (list, tuple)):
        encoder = encoder[0]
    encoder.eval()

    data.reset(test=test, string='Performing Linear SVC... ')

    Zs = []
    Ys = []
    try:
        while True:
            data.next()
            X, Y = data.get_batch('images', 'targets')
            Z = encoder(X)
            Ys.append(Y.data.cpu().numpy())
            Zs.append(Z.data.cpu().numpy())

    except StopIteration:
        pass

    Y = np.concatenate(Ys, axis=0)
    Z = np.concatenate(Zs, axis=0)

    return Y, Z

def svm_routine_train(data, models, losses, results, viz, encoder_key='encoder'):
    global SVM

    Y, Z = collect_embeddings(data, models, encoder_key=encoder_key)

    new_svm, predicted = perform_svc(Z, Y)
    correct = 100. * (predicted == Y).sum() / Y.shape[0]
    results['SVC_accuracy'] = correct

    SVM = new_svm


def svm_routine_test(data, models, losses, results, viz, encoder_key='encoder'):
    Y, Z = collect_embeddings(data, models, encoder_key=encoder_key, test=True)

    new_svm, predicted = perform_svc(Z, Y, clf=SVM)
    correct = 100. * (predicted == Y).sum() / Y.shape[0]
    results['SVC_accuracy'] = correct

# Builders =============================================================================================================

def build_encoder(models, x_shape, dim_z, Encoder, use_topnet=False, dim_top=None, **encoder_args):
    logger.debug('Forming encoder with class {} and args: {}'.format(Encoder, encoder_args))
    encoder = Encoder(x_shape, dim_out=dim_z, **encoder_args)

    if use_topnet:
        topnet = FullyConnectedNet(dim_z, dim_h=dim_top[::-1], dim_out=dim_top, batch_norm=True)
        revnet = FullyConnectedNet(dim_top, dim_h=dim_top, dim_out=dim_z, batch_norm=True)
        encoder = [encoder, topnet, revnet]

    models.update(encoder=encoder)


def build_discriminator(models, dim_in, key='discriminator'):
    discriminator = FullyConnectedNet(dim_in, dim_h=[2048, 1028, 512], dim_out=1, layer_norm=False, batch_norm=False)
    models[key] = discriminator

# CORTEX ===============================================================================================================
# Must include `BUILD`, `TRAIN_ROUTINES`, and `DEFAULT_CONFIG`

def SETUP(model=None, data=None, routines=None, **kwargs):
    noise = routines.discriminator.noise
    noise_type = routines.discriminator.noise_type
    if noise_type in ('unitsphere', 'unitball'):
        noise = 'normal'
    data.noise_variables = dict(y=dict(dist=noise, size=model.dim_noise),
                                u=dict(dist='uniform', size=1))

    routines.encoder.noise_type = routines.discriminator.noise_type
    routines.encoder.measure = routines.discriminator.measure
    routines.discriminator.output_nonlin = routines.encoder.output_nonlin
    routines.penalty.output_nonlin = routines.encoder.output_nonlin
    routines.penalty.noise_type = routines.discriminator.noise_type


def BUILD(data, models, model_type='convnet', use_topnet=False, dim_noise=None, dim_embedding=None, encoder_args=None,
          decoder_args=None, add_supervision=False):
    global TRAIN_ROUTINES, FINISH_TRAIN_ROUTINES, FINISH_TEST_ROUTINES

    if not use_topnet:
        dim_embedding = dim_noise
        dim_d = dim_embedding
    else:
        dim_d = dim_embedding + dim_noise

    x_shape = data.get_dims('x', 'y', 'c')
    dim_l = data.get_dims('labels')

    encoder_args = encoder_args or {}
    encoder_args.update(batch_norm=True, layer_norm=False)

    Encoder, encoder_args = update_encoder_args(x_shape, model_type=model_type, encoder_args=encoder_args)
    Decoder, decoder_args = update_decoder_args(x_shape, model_type=model_type, decoder_args=decoder_args)

    build_encoder(models, x_shape, dim_embedding, Encoder, use_topnet=use_topnet, dim_top=dim_noise,
                  fully_connected_layers=[1028], **encoder_args)
    build_discriminator(models, dim_d)
    if add_supervision:
        build_extra_networks(models, x_shape, dim_embedding, dim_l, Decoder, **decoder_args)
        TRAIN_ROUTINES.update(nets=network_routine)
        #FINISH_TRAIN_ROUTINES.update(svm=svm_routine_train)
        #FINISH_TEST_ROUTINES.update(svm=svm_routine_test)


TRAIN_ROUTINES = dict(discriminator=discriminator_routine, penalty=penalty_routine, encoder=encoder_routine)
FINISH_TRAIN_ROUTINES = dict()
FINISH_TEST_ROUTINES = dict()

DEFAULT_CONFIG = dict(
    data=dict(batch_size=dict(train=64, test=1028), skip_last_batch=True),
    optimizer=dict(
        optimizer='Adam',
        learning_rate=dict(discriminator=1e-4, nets=1e-4, encoder=1e-4),
        updates_per_routine=dict(discriminator=1, nets=1, encoder=1)),
    model=dict(model_type='convnet', dim_embedding=64, dim_noise=64, encoder_args=None, use_topnet=False,
               add_supervision=False),
    routines=dict(discriminator=dict(measure='JSD', noise_type='hypercubes', noise='uniform'),
                  penalty=dict(penalty_amount=0.2),
                  encoder=dict(generator_loss_type='non-saturating', output_nonlin=False),
                  nets=dict(),
                  svm=dict()),
    train=dict(epochs=500, archive_every=10, save_on_lowest='losses.encoder')
)