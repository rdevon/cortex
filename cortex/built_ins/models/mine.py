'''Mutual information neural estimation

'''


from cortex.plugins import register_model
from cortex.built_ins.models.ali import ALIDiscriminator
from cortex.built_ins.models.gan import GAN, GradientPenalty



class MINE(ALIDiscriminator):
    '''Mutual information neural estimation (MINE).

    Estimates mutual information of two random variables.

    '''

    def __init__(self):
        super().__init__(contract=dict(nets=dict(discriminator='mine')))


    def routine(self, X, Z, X_m, Z_m, mine_measure='JSD'):
        '''

        Args:
            mine_measure: MINE measure.
                {GAN, JSD, KL, RKL (reverse KL), X2 (Chi^2), H2
                (squared Hellinger), DV (Donsker Varahdan KL), W1 (IPM)}

        '''

        super().routine(X, X_m, Z, Z_m, measure=mine_measure)


class GAN_MINE(GAN):
    '''GAN + MINE.

    A generative adversarial network trained with MI maximization.

    '''
    def __init__(self):
        super().__init__()
        self.mine = MINE()

    def routine(self, X, Z, X_m, Z_m):
        self.generator.routine(Z)

    def train_step(self, generator_updates=1, discriminator_updates=1):
        '''

        Args:
            generator_updates: Number of generator updates per step.
            discriminator_updates: Number of discriminator updates per step.

        '''

        for _ in range(discriminator_updates):
            self.data.next()
            inputs, Z = self.inputs('inputs', 'Z')
            generated = self.generator.generate(Z)
            self.discriminator.routine(inputs, generated.detach())
            self.optimizer_step()
            self.penalty.train_step()

        for _ in range(generator_updates):
            self.generator.train_step()

    def eval_step(self):
        self.data.next()

        inputs, Z = self.inputs('inputs', 'Z')
        generated = self.generator.generate(Z)
        self.discriminator.routine(inputs, generated)
        self.penalty.routine(auto_input=True)
        self.generator.routine(auto_input=True)

    def visualize(self, images, Z):
        self.add_image(images, name='ground truth')
        generated = self.generator.generate(Z)
        self.discriminator.visualize(images, generated)
        self.generator.visualize(Z)


register_model(GAN_MINE)
