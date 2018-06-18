'''Generative adversarial networks with various objectives and penalties.

'''

import math

from cortex.plugins import (register_plugin, BuildPlugin, ModelPlugin,
                            RoutinePlugin)
import torch
from torch import autograd
import torch.nn.functional as F

from .utils import log_sum_exp, update_decoder_args, update_encoder_args


def raise_measure_error(measure):
    supported_measures = ['GAN', 'JSD', 'X2', 'KL', 'RKL', 'DV', 'H2', 'W1']
    raise NotImplementedError(
        'Measure `{}` not supported. Supported: {}'
        .format(measure, supported_measures))


def get_positive_expectation(p_samples, measure):
    log_2 = math.log(2.)

    if measure == 'GAN':
        Ep = - F.softplus(-p_samples)
    elif measure == 'JSD':
        Ep = log_2 - F.softplus(- p_samples)
    elif measure == 'X2':
        Ep = p_samples ** 2
    elif measure == 'KL':
        Ep = p_samples + 1.
    elif measure == 'RKL':
        Ep = -torch.exp(-p_samples)
    elif measure == 'DV':
        Ep = p_samples
    elif measure == 'H2':
        Ep = 1. - torch.exp(-p_samples)
    elif measure == 'W1':
        Ep = p_samples
    else:
        raise_measure_error(measure)

    return Ep.mean()


def get_negative_expectation(q_samples, measure):
    log_2 = math.log(2.)

    if measure == 'GAN':
        Eq = F.softplus(-q_samples) + q_samples
    elif measure == 'JSD':
        Eq = F.softplus(-q_samples) + q_samples - log_2
    elif measure == 'X2':
        Eq = -0.5 * ((torch.sqrt(q_samples ** 2) + 1.) ** 2)
    elif measure == 'KL':
        Eq = torch.exp(q_samples)
    elif measure == 'RKL':
        Eq = q_samples - 1.
    elif measure == 'DV':
        Eq = log_sum_exp(q_samples, 0) - math.log(q_samples.size(0))
    elif measure == 'H2':
        Eq = torch.exp(q_samples) - 1.
    elif measure == 'W1':
        Eq = q_samples
    else:
        raise_measure_error(measure)

    return Eq.mean()


def get_boundary(samples, measure):
    if measure in ('GAN', 'JSD', 'KL', 'RKL', 'H2', 'DV'):
        b = samples ** 2
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
        raise NotImplementedError(
            'Generator loss type `{}` not supported. '
            'Supported: [None, non-saturating, boundary-seek]')


class DiscriminatorRoutine(RoutinePlugin):
    '''Routine for discriminating

    '''
    plugin_name = 'discriminator'
    plugin_nets = ['discriminator', 'generator']
    plugin_vars = ['real', 'noise']

    def run(self, measure: str='GAN'):
        '''

        Args:
            measure: GAN measure.
                {GAN, JSD, KL, RKL (reverse KL), X2 (Chi^2), H2 (squared
                Hellinger), DV (Donsker Varahdan KL), W1 (IPM)}

        '''
        discriminator = self.nets.discriminator
        generator = self.nets.generator
        Z = self.vars.noise
        X_P = self.vars.real
        X_Q = F.tanh(generator(Z).detach())

        P_samples = discriminator(X_P)
        Q_samples = discriminator(X_Q)

        E_pos = get_positive_expectation(P_samples, measure)
        E_neg = get_negative_expectation(Q_samples, measure)
        difference = E_pos - E_neg

        self.results.update(Scores=dict(Ep=P_samples.mean().item(),
                                        Eq=Q_samples.mean().item()))
        self.results['{} distance'.format(measure)] = difference.item()
        self.add_image(X_P, name='ground truth')
        self.add_histogram(dict(fake=Q_samples.view(-1).data,
                                real=P_samples.view(-1).data),
                           name='discriminator output')
        self.losses.discriminator = -difference


class PenaltyRoutine(RoutinePlugin):
    '''Routine for applying gradient penalty.

    '''
    plugin_name = 'gradient_penalty'
    plugin_nets = ['network']
    plugin_vars = ['inputs']

    def run(self, penalty_type: str='contractive', penalty_amount: float=0.5):
        '''

        Args:
            penalty_type: Gradient penalty type for the discriminator.
                {contractive}
            penalty_amount: Amount of gradient penalty for the discriminator.

        '''
        if penalty_type == 'contractive':
            inputs = self.vars.inputs
            penalty = self.contractive_penalty(
                self.nets.network, inputs, penalty_amount=penalty_amount)
        else:
            raise NotImplementedError(penalty_type)

        if penalty:
            self.losses.network = penalty
            key = self.name + '_' + penalty_type + '_' + 'penalty'
            self.results[key] = penalty.item()

    @staticmethod
    def _get_gradient(inp, output):
        gradient = autograd.grad(outputs=output, inputs=inp,
                                 grad_outputs=torch.ones_like(output),
                                 create_graph=True, retain_graph=True,
                                 only_inputs=True)[0]
        return gradient

    def contractive_penalty(self, network, input, penalty_amount=0.5):

        if penalty_amount == 0.:
            return

        if not isinstance(input, (list, tuple)):
            input = [input]

        input = [inp.detach() for inp in input]
        input = [inp.requires_grad_() for inp in input]

        with torch.set_grad_enabled(True):
            output = network(*input)
        gradient = self._get_gradient(input, output)
        gradient = gradient.view(gradient.size()[0], -1)
        penalty = (gradient ** 2).sum(1).mean()

        return penalty_amount * penalty

    def interpolate_penalty(self, network, input, penalty_amount=0.5):

        input = input.detach()
        input = input.requires_grad_()

        if len(input) != 2:
            raise ValueError('tuple of 2 inputs required to interpolate')
        inp1, inp2 = input

        try:
            epsilon = network.inputs.e.view(-1, 1, 1, 1)
        except AttributeError:
            raise ValueError('You must initiate a uniform random variable'
                             '`e` to use interpolation')
        mid_in = ((1. - epsilon) * inp1 + epsilon * inp2)
        mid_in.requires_grad_()

        with torch.set_grad_enabled(True):
            mid_out = network(mid_in)
        gradient = self._get_gradient(mid_in, mid_out)
        gradient = gradient.view(gradient.size()[0], -1)
        penalty = ((gradient.norm(2, dim=1) - 1.) ** 2).mean()

        return penalty_amount * penalty


class GeneratorRoutine(RoutinePlugin):
    '''Routine for training a generator in GANs.

    '''
    plugin_name = 'generator'
    plugin_nets = ['generator', 'discriminator']
    plugin_vars = ['noise', 'generated']

    def run(self, measure: str=None, loss_type: str='non-saturating'):
        '''

        Args:
            loss_type: Generator loss type.
                {non-saturating, minimax, boundary-seek}

        '''
        Z = self.vars.noise
        discriminator = self.nets.discriminator
        generator = self.nets.generator

        X_Q = generator(Z)
        X_Q = F.tanh(X_Q)
        samples = discriminator(X_Q)

        g_loss = generator_loss(samples, measure, loss_type=loss_type)
        weights = get_weight(samples, measure)

        self.losses.generator = g_loss
        if weights is not None:
            self.results.update(Weights=weights.mean().item())
        self.add_image(X_Q, name='generated')

        self.vars.generated = X_Q


class DiscriminatorBuild(BuildPlugin):
    '''Build for the discriminator.

    '''
    plugin_name = 'discriminator'
    plugin_nets = ['discriminator']

    def build(self, discriminator_type: str='convnet', discriminator_args={}):
        '''

        Args:
            discriminator_type: Discriminator network type.
            discriminator_args: Discriminator network arguments.

        '''

        x_shape = self.get_dims('x', 'y', 'c')
        Encoder, discriminator_args = update_encoder_args(
            x_shape, model_type=discriminator_type,
            encoder_args=discriminator_args)
        discriminator = Encoder(x_shape, dim_out=1, **discriminator_args)
        self.nets.discriminator = discriminator


class GeneratorBuild(BuildPlugin):
    '''Build for the generator.

    '''
    plugin_name = 'generator'
    plugin_nets = ['generator']

    def build(self, generator_noise_type='normal', dim_z=64,
              generator_type: str='convnet', generator_args={}):
        '''

        Args:
            generator_noise_type: Type of input noise for the generator.
            dim_z: Input noise dimension for generator.
            generator_type: Generator network type.
            generator_args: Generator network arguments.

        '''
        x_shape = self.get_dims('x', 'y', 'c')
        self.add_noise('z', dist=generator_noise_type, size=dim_z)
        self.add_noise('e', dist='uniform', size=1)

        Decoder, generator_args = update_decoder_args(
            x_shape, model_type=generator_type, decoder_args=generator_args)
        generator = Decoder(x_shape, dim_in=dim_z, **generator_args)

        self.nets.generator = generator


class GAN(ModelPlugin):
    '''Generative adversarial network.

    A generative adversarial network on images.

    '''
    plugin_name = 'GAN'

    data_defaults = dict(batch_size=dict(train=64, test=64))
    train_defaults = dict(save_on_lowest='losses.gan')

    def __init__(self):
        super().__init__()
        self.builds.discriminator = DiscriminatorBuild()
        self.builds.generator = GeneratorBuild()
        self.routines.generator = GeneratorRoutine(noise='data.z')
        self.routines.discriminator = DiscriminatorRoutine(real='data.images',
                                                           noise='data.z')
        self.routines.penalty = PenaltyRoutine(network='discriminator',
                                               inputs='data.images')
        self.add_train_procedure(self.routines.generator,
                                 self.routines.discriminator,
                                 self.routines.penalty)
        self.add_eval_procedure(self.routines.generator,
                                self.routines.discriminator)


register_plugin(GAN)
