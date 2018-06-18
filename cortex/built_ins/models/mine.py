'''Mutual information neural estimation

'''


from cortex.plugins import ModelPlugin, register_plugin
from cortex.built_ins.models.ali import (ALIDiscriminatorBuild,
                                         ALIDiscriminatorRoutine)
from cortex.built_ins.models.gan import (GeneratorBuild, DiscriminatorBuild,
                                         DiscriminatorRoutine,
                                         GeneratorRoutine)


class MineDiscriminatorRoutine(ALIDiscriminatorRoutine):
    '''Routine for training MINE.

    '''
    plugin_name = 'MINE_discriminator'
    plugin_nets = ['generator', 'encoder', 'discriminator']
    plugin_vars = ['noise', 'noise_ind']

    def run(self, measure='DV'):
        '''

        Args:
            measure: MINE measure.
                {GAN, JSD, KL, RKL (reverse KL), X2 (Chi^2), H2
                (squared Hellinger), DV (Donsker Varahdan KL), W1 (IPM)}

        '''
        Z = self.vars.noise
        Z_Q = self.vars.noise_ind

        generator = self.nets.generator

        X_P = GeneratorRoutine.generate(generator, Z)
        X_Q = GeneratorRoutine.generate(generator, Z_Q)

        self.get_results(X_P, X_Q, Z, Z, measure)


class MINEGeneratorRoutine(GeneratorRoutine):

    '''Routine for training a generator in GANs with MINE regularization.

    '''
    plugin_name = 'mine_generator'
    plugin_nets = ['generator', 'discriminator', 'mine_discriminator']
    plugin_vars = ['noise', 'noise_ind', 'generated']

    def run(self, measure: str=None, loss_type: str='non-saturating',
            mine_measure='DV'):
        '''

        Args:
            loss_type: Generator loss type.
                {non-saturating, minimax, boundary-seek}

        '''
        super().run(measure=measure, loss_type=loss_type)

        Z = self.vars.noise
        Z_Q = self.vars.noise_ind
        X_P = self.vars.generated

        mine_discriminator = self.nets.mine_discriminator
        generator = self.nets.generator

        X_Q = self.generate(generator, Z_Q)

        E_pos, E_neg, _, _ = ALIDiscriminatorRoutine.score(
            mine_discriminator, X_P, X_Q, Z, Z, measure=mine_measure)
        distance = E_pos + E_neg

        self.losses.generator += -distance


class GAN_MINE(ModelPlugin):
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
        self.routines.generator = MINEGeneratorRoutine(noise='data.z', noise_ind='data.u')
        self.routines.discriminator = DiscriminatorRoutine(real='data.images',
                                                           noise='data.z')
        self.routines.penalty = PenaltyRoutine(network='discriminator',
                                               inputs='data.images')
        self.add_train_procedure(self.routines.generator,
                                 self.routines.discriminator,
                                 self.routines.penalty)
        self.add_eval_procedure(self.routines.generator,
                                self.routines.discriminator)


register_plugin(GAN_MINE)
