'''Adversarial autoencoder

'''

from cortex.plugins import ModelPlugin, register_model
from cortex.built_ins.models.gan import (SimpleDiscriminator, GradientPenalty,
                                         generator_loss)
from cortex.built_ins.models.vae import ImageDecoder, ImageEncoder


class AdversarialAutoencoder(ModelPlugin):
    '''Adversarial Autoencoder

    Autoencoder with a GAN loss on the latent space.

    '''

    defaults = dict(
        data=dict(batch_size=dict(train=64, test=640),
                  inputs=dict(inputs='images')),
        optimizer=dict(optimizer='Adam', learning_rate=1e-4),
        train=dict(epochs=500, archive_every=10)
    )

    def __init__(self):
        super().__init__()
        encoder_contract = dict(kwargs=dict(dim_out='dim_z'))
        decoder_contract = dict(kwargs=dict(dim_in='dim_z'))
        disc_contract = dict(kwargs=dict(dim_in='dim_z'))
        penalty_contract = dict(nets=dict(network='discriminator'))

        self.encoder = ImageEncoder(contract=encoder_contract)
        self.decoder = ImageDecoder(contract=decoder_contract)
        self.discriminator = SimpleDiscriminator(contract=disc_contract)
        self.penalty = GradientPenalty(contract=penalty_contract)

    def build(self, noise_type='normal', dim_z=64):
        '''

        Args:
            noise_type: Prior noise distribution.
            dim_z: Dimensionality of latent space.

        '''

        self.add_noise('Z', dist=noise_type, size=dim_z)
        self.encoder.build()
        self.decoder.build()
        self.discriminator.build()

    def routine(self, inputs, Z, encoder_loss_type='non-saturating',
                measure=None, beta=1.0):
        '''

        Args:
            encoder_loss_type: Adversarial loss type for the encoder.
            beta: Amount of adversarial loss for the encoder.

        '''

        Z_Q = self.encoder.encode(inputs)
        self.decoder.routine(inputs, Z_Q)

        E_pos, E_neg, P_samples, Q_samples = self.discriminator.score(
            Z, Z_Q, measure)

        adversarial_loss = generator_loss(
            Q_samples, measure, loss_type=encoder_loss_type)

        self.losses.encoder = self.losses.decoder + beta * adversarial_loss
        self.results.adversarial_loss = adversarial_loss.item()

    def train_step(self, n_discriminator_updates=1):
        '''

        Args:
            n_discriminator_updates: Number of discriminator updates per step.

        '''
        for _ in range(n_discriminator_updates):
            self.data.next()
            inputs, Z = self.inputs('inputs', 'Z')
            Z_Q = self.encoder.encode(inputs)
            self.discriminator.routine(Z, Z_Q)
            self.optimizer_step()
            self.penalty.routine(Z)
            self.optimizer_step()

        self.routine(auto_input=True)
        self.optimizer_step()

    def eval_step(self):
        self.data.next()
        inputs, Z = self.inputs('inputs', 'Z')
        Z_Q = self.encoder.encode(inputs)
        self.discriminator.routine(Z, Z_Q)
        self.penalty.routine(Z)

        self.routine(auto_input=True)

    def visualize(self, inputs, Z, targets):
        self.decoder.visualize(Z)
        self.encoder.visualize(inputs, targets)

        Z_Q = self.encoder.encode(inputs)
        R = self.decoder.decode(Z_Q)
        self.add_image(inputs, name='ground truth')
        self.add_image(R, name='reconstructed')


register_model(AdversarialAutoencoder)
