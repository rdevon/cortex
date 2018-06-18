'''Mutual information neural estimation

'''


from cortex.plugins import ModelPlugin, register_plugin, RoutinePlugin
from cortex.built_ins.models.ali import (ALIDiscriminatorBuild,
                                         ALIDiscriminatorRoutine,
                                         NoiseEncoderBuild)
from cortex.built_ins.models.gan import (GeneratorBuild, DiscriminatorBuild,
                                         DiscriminatorRoutine,
                                         GeneratorRoutine, PenaltyRoutine)
from cortex.built_ins.models.vae import ImageEncoderBuild


class MineDiscriminatorRoutine(ALIDiscriminatorRoutine):
    '''Routine for training MINE.

    '''
    plugin_name = 'MINE_discriminator'
    plugin_nets = ['generator', 'discriminator']
    plugin_vars = ['noise', 'noise_ind', 'targets']

    def run(self, mine_measure='JSD'):
        '''

        Args:
            mine_measure: MINE measure.
                {GAN, JSD, KL, RKL (reverse KL), X2 (Chi^2), H2
                (squared Hellinger), DV (Donsker Varahdan KL), W1 (IPM)}

        '''
        Z = self.vars.noise
        Z_Q = self.vars.noise_ind

        generator = self.nets.generator

        X_P = generator(Z)
        X_Q = generator(Z_Q)

        self.get_results(X_P, X_Q, Z, Z, mine_measure)


class MINEGeneratorRoutine(RoutinePlugin):

    '''Routine for training a generator in GANs with MINE regularization.

    '''
    plugin_name = 'mine_generator'
    plugin_nets = ['generator', 'mine_discriminator']
    plugin_vars = ['noise', 'noise_ind']

    def run(self, mine_measure=None):
        Z = self.vars.noise
        Z_Q = self.vars.noise_ind

        mine_discriminator = self.nets.mine_discriminator
        generator = self.nets.generator

        X_P = generator(Z)
        X_Q = generator(Z_Q)

        E_pos, E_neg, _, _ = ALIDiscriminatorRoutine.score(
            mine_discriminator, X_P, X_Q, Z, Z, measure=mine_measure)
        distance = E_pos + E_neg

        self.losses.generator = -distance


class GAN_MINE(ModelPlugin):
    '''Generative adversarial network.

    A generative adversarial network on images.

    '''
    plugin_name = 'GAN_MINE'

    data_defaults = dict(batch_size=dict(train=64, test=64))
    train_defaults = dict(save_on_lowest='losses.gan')

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

        self.builds.discriminator = DiscriminatorBuild()
        self.builds.generator = GeneratorBuild()
        self.builds.mine = ALIDiscriminatorBuild(
            discriminator='mine_discriminator')

        self.routines.generator = GeneratorRoutine(noise='data.z')
        self.routines.mine_generator = MINEGeneratorRoutine(
            noise='data.z', noise_ind='data.u')

        self.routines.discriminator = DiscriminatorRoutine(
            real='data.images', noise='data.z')
        self.routines.mine = MineDiscriminatorRoutine(
            discriminator='mine_discriminator', noise='data.z',
            noise_ind='data.u', targets='data.targets')
        self.routines.penalty = PenaltyRoutine(
            network='discriminator', inputs='data.images')
        self.routines.mine_penalty = PenaltyRoutine(
            network='mine_discriminator', inputs=('generated', 'data.z'))

        self.add_train_procedure(self.routines.generator,
                                 self.routines.mine_generator,
                                 self.routines.discriminator,
                                 self.routines.penalty,
                                 self.routines.mine,
                                 self.routines.mine_penalty)
        self.add_eval_procedure(self.routines.generator,
                                self.routines.mine_generator,
                                self.routines.discriminator,
                                self.routines.mine)


register_plugin(GAN_MINE)
