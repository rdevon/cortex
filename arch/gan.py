'''Generative adversarial networks with various objectives and penalties.

'''

import math

import torch
from torch import autograd
import torch.nn.functional as F

from vae import update_decoder_args, update_encoder_args
from utils import log_sum_exp


def raise_measure_error(measure):
    supported_measures = ['GAN', 'JSD', 'X2', 'KL', 'RKL', 'DV', 'H2', 'W1']
    raise NotImplementedError('Measure `{}` not supported. Supported: {}'.format(measure, supported_measures))


def get_positive_expectation(p_samples, measure):
    log_2 = math.log(2.)

    if   measure == 'GAN': Ep = -F.softplus(-p_samples)
    elif measure == 'JSD': Ep = log_2 - F.softplus(-p_samples)
    elif measure == 'X2':  Ep = p_samples ** 2
    elif measure == 'KL':  Ep = p_samples + 1.
    elif measure == 'RKL': Ep = -torch.exp(-p_samples)
    elif measure == 'DV':  Ep = p_samples
    elif measure == 'H2':  Ep = 1. - torch.exp(-p_samples)
    elif measure == 'W1':  Ep = p_samples
    else:
        raise_measure_error(measure)

    return Ep.mean()


def get_negative_expectation(q_samples, measure):
    log_2 = math.log(2.)

    if   measure == 'GAN': Eq = F.softplus(-q_samples) + q_samples
    elif measure == 'JSD': Eq = F.softplus(-q_samples) + q_samples - log_2
    elif measure == 'X2':  Eq = -0.5 * ((torch.sqrt(q_samples ** 2) + 1.) ** 2)
    elif measure == 'KL':  Eq = torch.exp(q_samples)
    elif measure == 'RKL': Eq = q_samples - 1.
    elif measure == 'DV':  Eq = log_sum_exp(q_samples, 0) - math.log(q_samples.size(0))
    elif measure == 'H2':  Eq = torch.exp(q_samples) - 1.
    elif measure == 'W1':  Eq = q_samples
    else:
        raise_measure_error(measure)

    return Eq.mean()


def get_boundary(samples, measure):
    if measure in ('GAN', 'JSD', 'KL', 'RKL', 'H2', 'DV'):
        b =samples ** 2
    elif measure == 'X2':
        b = (samples / 2.) ** 2
    elif measure == 'W':
        b = None
    else:
        raise_measure_error(measure)

    return b.mean()

def get_weight(samples, measure):
    if measure in ('GAN', 'JSD', 'KL', 'RKL', 'DV', 'H2'):
        return samples ** 2
    elif measure == 'X2':
        return (samples / 2.) ** 2
    elif measure == 'W1':
        return None
    else:
        raise_measure_error(measure)


def generator_loss(q_samples, measure, loss_type=None):
    if not loss_type or loss_type == 'minimax':
        return get_negative_expectation(q_samples, measure)
    elif loss_type == 'non-saturating':
        return -get_positive_expectation(q_samples, measure)
    elif loss_type == 'boundary-seek':
        return get_boundary(q_samples, measure)
    else:
        raise NotImplementedError('Generator loss type `{}` not supported. '
                                  'Supported: [None, non-saturating, boundary-seek]')


def apply_gradient_penalty(data, models, inputs=None, model=None, penalty_type='gradient_norm', penalty_amount=1.0):

    if penalty_amount == 0.:
        return
    if inputs is None:
        raise ValueError('No inputs provided')
    if not model:
        raise ValueError('No model provided')

    model_ = models[model]
    if not isinstance(inputs, (list, tuple)):
        inputs = [inputs]

    inputs = [inp.detach() for inp in inputs]
    inputs = [inp.requires_grad_() for inp in inputs]

    def get_gradient(inp, output):
        gradient = autograd.grad(outputs=output, inputs=inp, grad_outputs=torch.ones_like(output),
                                 create_graph=True, retain_graph=True, only_inputs=True)[0]
        return gradient

    if penalty_type == 'gradient_norm':
        penalties = []
        for inp in inputs:
            with torch.set_grad_enabled(True):
                output = model_(inp)
            gradient = get_gradient(inp, output)
            gradient = gradient.view(gradient.size()[0], -1)
            penalties.append((gradient ** 2).sum(1).mean())
        penalty = sum(penalties)

    elif penalty_type == 'interpolate':
        if len(inputs) != 2:
            raise ValueError('tuple of 2 inputs required to interpolate')
        inp1, inp2 = inputs

        try:
            epsilon = data['e'].view(-1, 1, 1, 1)
        except:
            raise ValueError('You must initiate a uniform random variable `e` to use interpolation')
        mid_in = ((1. - epsilon) * inp1 + epsilon * inp2)
        mid_in.requires_grad_()

        with torch.set_grad_enabled(True):
            mid_out = model_(mid_in)
        gradient = get_gradient(mid_in, mid_out)
        gradient = gradient.view(gradient.size()[0], -1)
        penalty = ((gradient.norm(2, dim=1) - 1.) ** 2).mean()

    else:
        raise NotImplementedError('Unsupported penalty {}'.format(penalty_type))

    return penalty_amount * penalty


# ROUTINES =============================================================================================================
# Each of these methods needs to take `data`, `models`, `losses`, `results`, and `viz`

def discriminator_routine(data, models, losses, results, viz, measure='GAN'):
    Z, X_P = data.get_batch('z', 'images')
    discriminator = models.discriminator
    generator = models.generator

    X_Q = generator(Z, nonlinearity=F.tanh).detach()
    P_samples = discriminator(X_P)
    Q_samples = discriminator(X_Q)

    E_pos = get_positive_expectation(P_samples, measure)
    E_neg = get_negative_expectation(Q_samples, measure)
    difference = E_pos - E_neg

    results.update(Scores=dict(Ep=P_samples.mean().item(), Eq=Q_samples.mean().item()))
    results['{} distance'.format(measure)] = difference.item()
    viz.add_image(X_P, name='ground truth')
    viz.add_histogram(dict(fake=Q_samples.view(-1).data, real=P_samples.view(-1).data), name='discriminator output')
    losses.discriminator = -difference


def penalty_routine(data, models, losses, results, viz, penalty_type='gradient_norm', penalty_amount=0.5):
    Z, X_P = data.get_batch('z', 'images')
    generator = models.generator

    X_Q = generator(Z, nonlinearity=F.tanh).detach()
    penalty = apply_gradient_penalty(data, models, inputs=(X_P, X_Q), model='discriminator',
                                     penalty_type=penalty_type, penalty_amount=penalty_amount)

    if penalty:
        losses.discriminator = penalty


def generator_routine(data, models, losses, results, viz, measure=None, loss_type='non-saturating'):
    Z = data['z']
    discriminator = models.discriminator
    generator = models.generator

    X_Q = generator(Z, nonlinearity=F.tanh)
    samples = discriminator(X_Q)

    g_loss = generator_loss(samples, measure, loss_type=loss_type)
    weights = get_weight(samples, measure)

    losses.generator = g_loss
    if weights is not None:
        results.update(Weights=weights.mean().item())
    viz.add_image(X_Q, name='generated')

# CORTEX ===============================================================================================================
# Must include `BUILD` and `TRAIN_ROUTINES`

def BUILD(data, models, model_type='convnet', discriminator_args=dict(), generator_args=dict()):
    x_shape = data.get_dims('x', 'y', 'c')
    dim_z = data.get_dims('z')

    Encoder, discriminator_args = update_encoder_args(x_shape, model_type=model_type, encoder_args=discriminator_args)
    Decoder, generator_args = update_decoder_args(x_shape, model_type=model_type, decoder_args=generator_args)

    discriminator = Encoder(x_shape, dim_out=1, **discriminator_args)
    generator = Decoder(x_shape, dim_in=dim_z, **generator_args)

    models.update(generator=generator, discriminator=discriminator)


def SETUP(routines=None, **kwargs):
    routines.generator.measure = routines.discriminator.measure


TRAIN_ROUTINES = dict(discriminator=discriminator_routine, penalty=penalty_routine, generator=generator_routine)

INFO = dict(measure=dict(choices=['GAN', 'JSD', 'KL', 'RKL', 'X2', 'H2', 'DV', 'W1'],
                         help='GAN measure. {GAN, JSD, KL, RKL (reverse KL), X2 (Chi^2), H2 (squared Hellinger), '
                              'DV (Donsker Varahdan KL), W1 (IPM)}'),
            loss_type=dict(choices=['non-saturating', 'minimax', 'boundary-seek'],
                           help='Generator loss type.'),
            penalty_type=dict(choices=['gradient_norm', 'interpolate'],
                              help='Gradient penalty type for the discriminator.'),
            penalty_amount=dict(help='Amount of gradient penalty for the discriminator.'),
            model_type=dict(choices=['mnist', 'convnet', 'resnet'],
                            help='Model type.')
)

TEST_ROUTINES = dict(penalty=None)
DEFAULT_CONFIG = dict(
    data=dict(batch_size=dict(train=64, test=1000),
              noise_variables=dict(z=dict(dist='normal', size=64, loc=0, scale=1),
                                   e=dict(dist='uniform', size=1, low=0, high=1))),
    train=dict(save_on_lowest='losses.gan')
)