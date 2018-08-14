'''Adversarially learned inference and Bi-GAN

Currently noise encoder is not implemented.

'''

__author__ = 'R Devon Hjelm and Samuel Lavoie'
__author_email__ = 'erroneus@gmail.com'

from cortex.plugins import register_plugin, ModelPlugin
from cortex.built_ins.models.gan import (
    get_positive_expectation, get_negative_expectation, GradientPenalty)
from cortex.built_ins.models.vae import ImageDecoder, ImageEncoder
from cortex.built_ins.networks.fully_connected import FullyConnectedNet

import torch
import torch.nn as nn


class ALIDiscriminatorModule(nn.Module):
    '''ALI discriminator model.

    '''
    def __init__(self, x_encoder, z_encoder, topnet):
        super(ALIDiscriminatorModule, self).__init__()
        self.x_encoder = x_encoder
        self.z_encoder = z_encoder
        self.topnet = topnet

    def forward(self, x, z, nonlinearity=None):
        w = self.x_encoder(x)

        if len(w.size()) == 4:
            w = w.view(-1, w.size(1) * w.size(2) * w.size(3))

        if self.z_encoder is None:
            v = z
        else:
            v = self.z_encoder(z)
        v = torch.cat([v, w], dim=1)
        y = self.topnet(v)

        return y


class BidirectionalModel(ModelPlugin):

    def __init__(self, discriminator, **kwargs):
        super().__init__(**kwargs)
        self.discriminator = discriminator

        encoder_contract = dict(kwargs=dict(dim_out='dim_z'))
        decoder_contract = dict(kwargs=dict(dim_in='dim_z'))
        self.decoder = ImageDecoder(contract=decoder_contract)
        self.encoder = ImageEncoder(contract=encoder_contract)

    def build(self):
        self.decoder.build()
        self.encoder.build()

    def routine(self, inputs, Z, measure=None):
        decoder = self.nets.decoder
        encoder = self.nets.encoder

        X_P, Z_Q = inputs, Z

        X_Q = decoder(Z_Q)
        Z_P = encoder(X_P)

        E_pos, E_neg, _, _ = self.discriminator.score(
            X_P, X_Q, Z_P, Z_Q, measure)

        self.losses.decoder = -E_neg
        self.losses.encoder = E_pos

    def visualize(self):
        self.decoder.visualize(auto_input=True)
        self.encoder.visualize(auto_input=True)


class ALIDiscriminator(ModelPlugin):

    def build(self, topnet_args=dict(dim_h=[512, 128], batch_norm=False),
              dim_int=256, dim_z=None):
        '''
        Args:
            topnet_args: Keyword arguments for the top network.
            dim_int: Intermediate layer size for discriminator.
        '''

        x_encoder = self.nets.x_encoder

        try:
            z_encoder = self.nets.z_encoder
        except KeyError:
            z_encoder = None

        if z_encoder is not None:
            z_encoder_out = list(
                z_encoder.models[-1].parameters())[-1].size()[0]
            dim_in = dim_int + z_encoder_out
        else:
            dim_in = dim_int + dim_z

        topnet = FullyConnectedNet(dim_in, 1, **topnet_args)

        discriminator = ALIDiscriminatorModule(x_encoder, z_encoder, topnet)

        self.nets.discriminator = discriminator

    def routine(self, X_real, X_fake, Z_real, Z_fake, measure='GAN'):
        '''

        Args:
            measure: GAN measure.
                {GAN, JSD, KL, RKL (reverse KL), X2 (Chi^2), H2
                (squared Hellinger), DV (Donsker Varahdan KL), W1 (IPM)}

        '''

        X_P, Z_P = X_real, Z_real
        X_Q, Z_Q = X_fake, Z_fake

        E_pos, E_neg, P_samples, Q_samples = self.score(
            X_P, X_Q, Z_P, Z_Q, measure)
        difference = E_pos - E_neg

        self.losses.discriminator = -difference
        self.results.update(Scores=dict(Ep=P_samples.mean().item(),
                                        Eq=Q_samples.mean().item()))
        self.results['{} distance'.format(measure)] = difference.item()

    def score(self, X_P, X_Q, Z_P, Z_Q, measure):
        discriminator = self.nets.discriminator

        P_samples = discriminator(X_P, Z_P)
        Q_samples = discriminator(X_Q, Z_Q)

        E_pos = get_positive_expectation(P_samples, measure)
        E_neg = get_negative_expectation(Q_samples, measure)

        return E_pos, E_neg, P_samples, Q_samples

    def visualize(self, X_real, X_fake, Z_real, Z_fake, targets):
        discriminator = self.nets.discriminator

        X_P, Z_P = X_real, Z_real
        X_Q, Z_Q = X_fake, Z_fake

        P_samples = discriminator(X_P, Z_P)
        Q_samples = discriminator(X_Q, Z_Q)

        self.add_histogram(dict(fake=Q_samples.view(-1).data,
                                real=P_samples.view(-1).data),
                           name='discriminator output')
        self.add_scatter(Z_P, labels=targets.data, name='latent values')


class ALI(ModelPlugin):
    '''Adversarially learned inference.

    a.k.a. BiGAN
    Note:
        Currently noisy encoder not supported.

    '''

    defaults = dict(
        data=dict(batch_size=dict(train=64, test=640),
                  inputs=dict(inputs='images')),
        optimizer=dict(optimizer='Adam', learning_rate=1e-4),
        train=dict(epochs=500, save_on_lowest='losses.generator')
    )

    def __init__(self):
        super().__init__()
        self.discriminator = ALIDiscriminator()
        self.bidirectional_model = BidirectionalModel(
            discriminator=self.discriminator)

        encoder_contract = dict(nets=dict(encoder='x_encoder'),
                                kwargs=dict(dim_out='dim_int'))
        self.encoder = ImageEncoder(contract=encoder_contract)

        penalty_contract = dict(nets=dict(network='discriminator'))
        self.penalty = GradientPenalty(contract=penalty_contract)

    def build(self, dim_z=None, dim_int=None, use_z_encoder=False,
              z_encoder_args=dict(dim_h=256, batch_norm=True),
              noise_type='normal'):
        '''

        Args:
            use_z_encoder: Use a neural network for Z pathway in discriminator.
            z_encoder_args: Arguments for the Z pathway encoder.

        '''
        self.add_noise('Z', dist=noise_type, size=dim_z)

        if use_z_encoder:
            encoder = FullyConnectedNet(dim_z, dim_int, **z_encoder_args)
            self.nets.z_encoder = encoder

        self.encoder.build()
        self.discriminator.build()
        self.bidirectional_model.build()

    def train_step(self, discriminator_updates=1):
        '''

        Args:
            generator_updates: Number of generator updates per step.
            discriminator_updates: Number of discriminator updates per step.

        '''
        for _ in range(discriminator_updates):
            self.data.next()
            inputs, Z = self.inputs('inputs', 'Z')

            generated = self.bidirectional_model.decoder.decode(Z)
            inferred = self.bidirectional_model.encoder.encode(inputs)

            self.discriminator.routine(
                inputs, generated.detach(), inferred.detach(), Z)
            self.optimizer_step()

            self.penalty.routine((inputs, inferred))
            self.optimizer_step()

        self.bidirectional_model.train_step()

    def eval_step(self):
        self.data.next()
        inputs, Z = self.inputs('inputs', 'Z')

        generated = self.bidirectional_model.decoder.decode(Z)
        inferred = self.bidirectional_model.encoder.encode(inputs)

        self.discriminator.routine(inputs, generated, inferred, Z)
        self.bidirectional_model.eval_step()

    def visualize(self, inputs, Z, targets):
        generated = self.bidirectional_model.decoder.decode(Z)
        inferred = self.bidirectional_model.encoder.encode(inputs)

        self.bidirectional_model.visualize()
        self.discriminator.visualize(inputs, generated, inferred, Z, targets)


register_plugin(ALI)
