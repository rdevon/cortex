'''Adversarially learned inference and Bi-GAN

Currently noise encoder is not implemented.

'''

__author__ = 'R Devon Hjelm and Samuel Lavoie'
__author_email__ = 'erroneus@gmail.com'

from cortex.plugins import (register_plugin, BuildPlugin, ModelPlugin,
                            RoutinePlugin)
from cortex.built_ins.models.gan import (get_positive_expectation,
                                         get_negative_expectation,
                                         GeneratorBuild, PenaltyRoutine)
from cortex.built_ins.models.vae import ImageEncoderBuild
from cortex.built_ins.networks.fully_connected import FullyConnectedNet
import torch
import torch.nn as nn
import torch.nn.functional as F


class ALIDiscriminator(nn.Module):
    '''ALI discriminator model.
    '''

    def __init__(self, x_encoder, z_encoder, topnet):
        super(ALIDiscriminator, self).__init__()
        self.x_encoder = x_encoder
        self.z_encoder = z_encoder
        self.topnet = topnet

    def forward(self, x, z, nonlinearity=None):
        w = F.relu(self.x_encoder(x))
        if self.z_encoder is None:
            v = torch.cat([w, z], 1)
        else:
            u = F.relu(self.z_encoder(z))
            v = torch.cat([w, u], 1)
        y = self.topnet(v, nonlinearity=nonlinearity)
        return y


class ALIDiscriminatorRoutine(RoutinePlugin):
    '''Adversarially-learned inference / BiGAN

    '''
    plugin_name = 'ALI_discriminator'
    plugin_nets = ['generator', 'encoder', 'discriminator']
    plugin_vars = ['real', 'noise', 'targets']

    def run(self, measure='GAN'):
        '''

        Args:
            measure: GAN measure.
                {GAN, JSD, KL, RKL (reverse KL), X2 (Chi^2), H2
                (squared Hellinger), DV (Donsker Varahdan KL), W1 (IPM)}

        '''
        X_P = self.vars.real
        T = self.vars.targets
        Z_Q = self.vars.noise
        generator = self.nets.generator
        encoder = self.nets.encoder
        discriminator = self.nets.discriminator

        X_Q = F.tanh(generator(Z_Q).detach())
        Z_P = encoder(X_P).detach()

        E_pos, E_neg, P_samples, Q_samples = self.score(
            discriminator, X_P, X_Q, Z_P, Z_Q, measure)
        difference = E_pos - E_neg

        self.losses.discriminator = -difference

        self.results.update(Scores=dict(Ep=P_samples.mean().item(),
                                        Eq=Q_samples.mean().item()))
        self.results['{} distance'.format(measure)] = difference.item()
        self.add_image(X_Q, name='generated')
        self.add_image(X_P, name='ground truth')
        self.add_histogram(dict(fake=Q_samples.view(-1).data,
                                real=P_samples.view(-1).data),
                           name='discriminator output')
        self.add_scatter(Z_P, labels=T.data, name='latent values')

    @staticmethod
    def score(discriminator, X_P, X_Q, Z_P, Z_Q, measure):
        P_samples = discriminator(X_P, Z_P)
        Q_samples = discriminator(X_Q, Z_Q)

        E_pos = get_positive_expectation(P_samples, measure)
        E_neg = get_negative_expectation(Q_samples, measure)

        return E_pos, E_neg, P_samples, Q_samples


class ALIGeneratorRoutine(RoutinePlugin):
    '''Routine for the encoder and decoder of ALI.

    '''
    plugin_name = 'ALI_generator'
    plugin_nets = ['generator', 'encoder', 'discriminator']
    plugin_vars = ['real', 'noise', 'generated', 'inferred']

    def run(self, measure=None):
        X_P = self.vars.real
        Z_Q = self.vars.noise
        generator = self.nets.generator
        encoder = self.nets.encoder
        discriminator = self.nets.discriminator

        X_Q = F.tanh(generator(Z_Q))
        Z_P = encoder(X_P)

        E_pos, E_neg, _, _ = ALIDiscriminatorRoutine.score(
            discriminator, X_P, X_Q, Z_P, Z_Q, measure)

        self.losses.generator = -E_neg
        self.losses.encoder = E_pos

        self.vars.generated = X_Q.detach()
        self.vars.inferred = Z_P.detach()


class DecoderRoutine(RoutinePlugin):
    '''Routine for a simple decoder for images.

    '''
    plugin_name = 'decoder'
    plugin_nets = ['decoder']
    plugin_vars = ['inputs', 'targets']

    def run(self):
        Z = self.vars.inputs
        X = self.vars.targets

        decoder = self.nets.decoder

        X_d = decoder(Z)
        X_d = F.tanh(X_d)
        self.losses.decoder = ((X - X_d) ** 2).sum(1).sum(1).sum(1).mean()

        self.add_image(X, name='Ground truth')
        self.add_image(X_d, name='Reconstruction')


class NoiseEncoderBuild(BuildPlugin):
    '''Builder for encoding the noise for discrimination.
    '''
    plugin_name = 'Noise_encoder'
    plugin_nets = ['encoder']

    def build(self, dim_in=None, dim_out=None,
              encoder_args=dict(dim_h=[64], batch_norm=False)):
        '''
        Args:
            dim_in: Input dimension.
            dim_out: Output dimension.
            encoder_args: Arguments for the encoder.
        '''

        encoder = FullyConnectedNet(dim_in, dim_out, **encoder_args)
        self.nets.encoder = encoder


class ALIDiscriminatorBuild(BuildPlugin):
    '''Builder for ALI discriminator.
    '''
    plugin_name = 'ALI_discriminator'
    plugin_nets = ['x_encoder', 'z_encoder', 'discriminator']

    def build(self, topnet_args=dict(dim_h=[512, 128], batch_norm=False),
              dim_int=256, dim_z=None):
        '''
        Args:
            topnet_args: Keyword arguments for the top network.
        '''

        x_encoder = self.nets.x_encoder
        z_encoder = self.nets.z_encoder

        if z_encoder is not None:
            dim_in = 2 * dim_int
        else:
            dim_in = dim_int + dim_z

        topnet = FullyConnectedNet(dim_in, 1, **topnet_args)

        discriminator = ALIDiscriminator(x_encoder, z_encoder, topnet)

        self.nets.discriminator = discriminator


class ALI(ModelPlugin):
    '''Adversarially learned inference.

    Note:
        Noise in the encoder is not implemented yet.

    '''
    plugin_name = 'ALI'

    data_defaults = dict(batch_size=dict(train=64, test=640))
    optimizer_defaults = dict(optimizer='Adam', learning_rate=1e-4)
    train_defaults = dict(epochs=500, save_on_lowest='losses.generator')

    def __init__(self, use_z_encoder=False):
        super().__init__()
        self.builds.x_encoder = ImageEncoderBuild(image_encoder='x_encoder',
                                                  dim_out='dim_int',
                                                  encoder_args='x_encoder_args')
        if use_z_encoder:
            self.builds.z_encoder = NoiseEncoderBuild(
                encoder='z_encoder', dim_in='dim_z', dim_out='dim_int',
                encoder_args='z_encoder_args')
        self.builds.encoder = ImageEncoderBuild(image_encoder='encoder',
                                                dim_out='dim_z')
        self.builds.discriminator = ALIDiscriminatorBuild()
        self.builds.generator = GeneratorBuild()

        self.routines.ali_discriminator = ALIDiscriminatorRoutine(
            real='data.images', noise='data.z', targets='data.targets')
        self.routines.ali_generator = ALIGeneratorRoutine(real='data.images',
                                                          noise='data.z')
        self.routines.penalty = PenaltyRoutine(network='discriminator',
                                               inputs=('data.images',
                                                       'inferred'))
        self.add_train_procedure(self.routines.ali_generator,
                                 self.routines.ali_discriminator,
                                 self.routines.penalty)
        self.add_eval_procedure(self.routines.ali_generator,
                                self.routines.ali_discriminator)


register_plugin(ALI)
