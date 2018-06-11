'''Adversarially learned inference and Bi-GAN

Currently noise encoder is not implemented.

'''

__author__ = 'R Devon Hjelm and Samuel Lavoie'
__author_email__ = 'erroneus@gmail.com'

from cortex.plugins import register_plugin, BuildPlugin, ModelPlugin, RoutinePlugin
from cortex.built_ins.models.gan import GeneratorBuild, PenaltyRoutine
from cortex.built_ins.models.vae import ImageDecoderBuild
import torch
import torch.nn as nn
import torch.nn.functional as F

from .gan import get_positive_expectation, get_negative_expectation


class ALIDiscriminator(nn.Model):
    '''ALI discriminator model.

    '''
    def __init__(self, x_encoder, z_encoder, topnet):
        super(ALIDiscriminator, self).__init__()
        self.x_encoder = x_encoder
        self.z_encoder = z_encoder
        self.topnet = topnet

    def forward(self, x, z, nonlinearity=None):
        w = F.relu(self.x_encoder(x))
        u = F.relu(self.z_encoder(z))
        y = self.topnet(torch.cat([w, u], 1), nonlinearity=nonlinearity)
        return y


class ALIDiscriminatorRoutine(RoutinePlugin):
    '''Adversarially-learned inference / BiGAN

    '''
    plugin_name = 'ALI_discriminator'
    plugin_nets = ['generator', 'decoder', 'discriminator']
    plugin_inputs = ['real', 'noise', 'targets']

    def run(self, measure='GAN'):
        '''

        Args:
            measure: GAN measure.
                {GAN, JSD, KL, RKL (reverse KL), X2 (Chi^2), H2 (squared Hellinger),
                DV (Donsker Varahdan KL), W1 (IPM)}

        '''
        X_P = self.inputs.real
        T = self.inputs.targets
        Z_Q = self.inputs.noise
        generator = self.nets.generator
        decoder = self.nets.decoder
        discriminator = self.nets.discriminator

        X_Q = decoder(Z_Q).detach()
        X_Q = F.tanh(X_Q)
        Z_P = generator(X_P).detach()

        E_pos, E_neg, P_samples, Q_samples = self.score(discriminator, X_P, X_Q, Z_P, Z_Q, measure)
        difference = E_pos - E_neg

        self.losses.discriminator = -difference

        self.results.update(Scores=dict(Ep=P_samples.mean().item(), Eq=Q_samples.mean().item()))
        self.results['{} distance'.format(measure)] = difference.item()
        self.add_image(X_Q, name='generated')
        self.add_histogram(dict(fake=Q_samples.view(-1).data, real=P_samples.view(-1).data),
                           name='discriminator output')
        self.add_scatter(Z_P, labels=T.data, name='latent values')

    @staticmethod
    def score(discriminator, X_P, X_Q, Z_P, Z_Q, measure):

        P_samples = discriminator(X_P, Z_P)
        Q_samples = discriminator(X_Q, Z_Q)

        E_pos = get_positive_expectation(P_samples, measure)
        E_neg = get_negative_expectation(Q_samples, measure)

        return E_pos, E_neg, P_samples, Q_samples
register_plugin(ALIDiscriminatorRoutine)


class ALIGeneratorRoutine(RoutinePlugin):
    '''Routine for the encoder and decoder of ALI.

    '''
    plugin_name = 'ALI_generator'
    plugin_nets = ['generator', 'decoder', 'discriminator']
    plugin_inputs = ['real', 'noise']
    plugin_outputs = ['generated', 'inferred']

    def run(self, measure=None):
        X_P = self.inputs.real
        Z_Q = self.inputs.noise
        generator = self.nets.generator
        decoder = self.nets.decoder
        discriminator = self.nets.discriminator

        X_Q = decoder(Z_Q)
        X_Q = F.tanh(X_Q)
        Z_P = generator(X_P)

        E_pos, E_neg, P_samples, Q_samples = ALIDiscriminatorRoutine.score(discriminator, X_P, X_Q, Z_P, Z_Q, measure)
        difference = E_pos - E_neg

        self.losses.generator = difference
register_plugin(ALIGeneratorRoutine)


class DecoderRoutine(RoutinePlugin):
    '''Routine for a simple decoder for images.

    '''
    plugin_name = 'decoder'
    plugin_nets = ['decoder']
    plugin_inputs = ['inputs', 'targets']

    def run(self):
        Z = self.inputs.inputs
        X = self.inputs.targets

        decoder = self.nets.decoder

        X_d = decoder(Z)
        X_d = F.tanh(X_d)
        self.losses.decoder = ((X - X_d) ** 2).sum(1).sum(1).sum(1).mean()

        self.add_image(X, name='Ground truth')
        self.add_image(X_d, name='Reconstruction')
register_plugin(DecoderRoutine)


class ALIDiscriminatorBuild(BuildPlugin):
    '''Builder for ALI discriminator.

    '''
    plugin_name = 'ALI_discriminator'
    plugin_nets = ['ali_discriminator']
    plugin_deps = ['x_encoder', 'z_encoder', 'topnet']

    def build(self):
        x_encoder = self.nets.x_encoder
        z_encoder = self.nets.z_encoder
        topnet = self.nets.topnet

        discriminator = ALIDiscriminator(x_encoder, z_encoder, topnet)

        self.nets.ali_discriminator = discriminator

register_plugin(ALIDiscriminatorBuild)


class ALI(ModelPlugin):
    '''Adversarially learned inference.

    Note:
        Noise in the encoder is not implemented yet.

    '''
    plugin_name = 'ALI'

    data_defaults = dict(batch_size=dict(train=64, test=640))
    optimizer_defaults = dict(optimizer='Adam', learning_rate=1e-4)
    train_defaults=dict(epochs=500, save_on_lowest='losses.generator')

    def __init__(self):
        super().__init__()
        self.add_build(ALIDiscriminatorBuild)
        self.add_build(GeneratorBuild)
        self.add_build(ImageDecoderBuild, dim_in='dim_z', image_decoder='decoder')

        self.add_routine(ALIDiscriminatorRoutine, real='data.images', noise='data.z', targets='data.targets')
        self.add_routine(ALIGeneratorRoutine, real='data.images', noise='data.z')
        self.add_routine(PenaltyRoutine, )