'''Mutual information neural estimation

'''


from cortex.plugins import register_model
from cortex.built_ins.models.ali import ALIDiscriminator
from cortex.built_ins.models.gan import GAN, GradientPenalty
from cortex.built_ins.models.vae import ImageEncoder


class MINE(ALIDiscriminator):
    '''Mutual information neural estimation (MINE).

    Estimates mutual information of two random variables.

    '''

    def __init__(self):
        super().__init__(contract=dict(nets=dict(discriminator='mine')))
        contract = dict(nets=dict(network='mine'),
                        kwargs=dict(penalty_type='mine_penalty_type',
                                    penalty_amount='mine_penalty_amount'))
        self.penalty = GradientPenalty(contract=contract)

    def routine(self, X, X_m, Z, Z_m, mine_measure='JSD'):
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

        encoder_contract = dict(nets=dict(encoder='x_encoder'),
                                kwargs=dict(dim_out='dim_int'))
        self.encoder = ImageEncoder(contract=encoder_contract)

    def build(self, noise_type='normal', dim_z=64):
        super().build(noise_type=noise_type, dim_z=dim_z)
        self.encoder.build()
        self.mine.build()

    def routine(self, Z, Z_m, mine_measure=None, beta=1.0):
        '''

        Args:
            beta: Factor for mutual information maximization for generator.

        '''
        self.generator.routine(Z)
        X = self.generator.generate(Z)
        E_pos, E_neg, _, _ = self.mine.score(X, X, Z, Z_m, mine_measure)

        self.losses.generator += (E_neg - E_pos)

    def train_step(self, mine_updates=1, discriminator_updates=1):
        '''

        Args:
            mine_updates: Number of MINE updates per step.
            discriminator_updates: Number of discriminator updates per step.

        '''

        for _ in range(discriminator_updates):
            self.data.next()
            inputs, Z = self.inputs('inputs', 'Z')
            generated = self.generator.generate(Z)
            self.discriminator.routine(inputs, generated.detach())
            self.optimizer_step()
            self.penalty.train_step()

        Z_P = Z

        for _ in range(mine_updates):
            self.data.next()
            Z = self.inputs('Z')
            generated = self.generator.generate(Z)
            self.mine.routine(generated, Z, generated, Z_P)
            self.optimizer_step()

        self.routine(Z, Z_P)
        self.optimizer_step()

    def eval_step(self):
        self.data.next()
        inputs, Z = self.inputs('inputs', 'Z')
        generated = self.generator.generate(Z)
        self.discriminator.routine(inputs, generated.detach())

        Z_P = Z

        self.data.next()
        Z = self.inputs('Z')
        generated = self.generator.generate(Z)
        self.mine.routine(generated, Z, generated, Z_P)

        self.routine(Z, Z_P)

    def visualize(self, images, Z, targets):
        self.add_image(images, name='ground truth')
        generated = self.generator.generate(Z)
        self.discriminator.visualize(images, generated)
        self.generator.visualize(Z)
        self.data.next()
        Z_N = self.inputs('Z')

        self.mine.visualize(generated, generated, Z, Z_N, targets)


register_model(GAN_MINE)
