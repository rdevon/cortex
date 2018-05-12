'''Training a generator my mapping an encoder to noise.

'''

import torch.nn.functional as F

from gan import apply_gradient_penalty
from featnet import build_discriminator, encode, score, shape_noise
from vae import update_encoder_args, update_decoder_args


def encoder_routine(data, models, losses, results, viz, measure=None, noise_type=None, output_nonlin=False,
                    offset=None):
    Z, Y_P, Y_Q, U, X_P = data.get_batch('z', 'y_p', 'y_q', 'u', 'images')
    Y_P = shape_noise(Y_P, U, noise_type)
    Y_Q = shape_noise(Y_Q, U, noise_type, offset=offset)

    generator = models.generator

    X_Q = generator(Z, nonlinearity=F.tanh).detach()
    _, W_P, _ = encode(models, X_P, None, output_nonlin=output_nonlin, noise_type=noise_type)
    _, W_Q, _ = encode(models, X_Q, None, output_nonlin=output_nonlin, noise_type=noise_type)

    # Real discriminator
    E_P_pos, E_P_neg, S_PP, S_PQ = score(models, Y_P, W_P, measure, key='real_discriminator')
    P_difference = E_P_pos - E_P_neg

    # Fake discriminator
    E_Q_pos, E_Q_neg, S_QP, S_QQ = score(models, Y_Q, W_Q, measure, key='fake_discriminator')
    Q_difference = E_Q_pos - E_Q_neg

    losses.encoder = P_difference + Q_difference
    results.update(Scores=dict(Epp=S_PP.mean().item(), Epq=S_PQ.mean().item(),
                               Eqp=S_QP.mean().item(), Eqq=S_QQ.mean().item()))
    results['{} distances'.format(measure)] = dict(P=P_difference.item(), Q=Q_difference.item())
    viz.add_image(X_P, name='ground truth')
    viz.add_histogram(dict(fake=S_PQ.view(-1).data, real=S_PP.view(-1).data), name='real discriminator output')
    viz.add_histogram(dict(fake=S_QQ.view(-1).data, real=S_QP.view(-1).data), name='fake discriminator output')


def discriminator_routine(data, models, losses, results, viz, measure='JSD', noise_type=None,
                          output_nonlin=None, offset=1.0):
    Z, Y_P, Y_Q, U, X_P = data.get_batch('z', 'y_p', 'y_q', 'u', 'images')
    Y_P = shape_noise(Y_P, U, noise_type)
    Y_Q = shape_noise(Y_Q, U, noise_type, offset=offset)

    generator = models.generator

    X_Q = generator(Z, nonlinearity=F.tanh).detach()
    _, W_P, _ = encode(models, X_P, None, output_nonlin=output_nonlin, noise_type=noise_type)
    _, W_Q, _ = encode(models, X_Q, None, output_nonlin=output_nonlin, noise_type=noise_type)

    # Real discriminator
    E_P_pos, E_P_neg, _, _ = score(models, Y_P, W_P, measure, key='real_discriminator')
    P_difference = E_P_pos - E_P_neg

    # Fake discriminator
    E_Q_pos, E_Q_neg, _, _ = score(models, Y_Q, W_Q, measure, key='fake_discriminator')
    Q_difference = E_Q_pos - E_Q_neg

    losses.real_discriminator = -P_difference
    losses.fake_discriminator = -Q_difference


def penalty_routine(data, models, losses, results, viz, penalty_amount=0.2, offset=None,
                    encoder_penalty_amount=0.5, output_nonlin=None, noise_type=None):
    Z, Y_P, Y_Q, U, X_P = data.get_batch('z', 'y_p', 'y_q', 'u', 'images')
    Y_P = shape_noise(Y_P, U, noise_type)
    Y_Q = shape_noise(Y_Q, U, noise_type, offset=offset)
    generator = models.generator

    X_Q = generator(Z, nonlinearity=F.tanh).detach()

    _, W_P, _ = encode(models, X_P, None, output_nonlin=output_nonlin, noise_type=noise_type)
    _, W_Q, _ = encode(models, X_Q, None, output_nonlin=output_nonlin, noise_type=noise_type)

    P_penalty = apply_gradient_penalty(data, models, inputs=(Y_P, W_P), model='real_discriminator',
                                       penalty_amount=penalty_amount)
    if P_penalty:
        losses.real_discriminator = P_penalty

    Q_penalty = apply_gradient_penalty(data, models, inputs=(Y_Q, W_Q), model='fake_discriminator',
                                       penalty_amount=penalty_amount)
    if Q_penalty:
        losses.fake_discriminator = Q_penalty

    E_penalty = apply_gradient_penalty(data, models, inputs=(X_P, X_Q), penalty_type='dot',
                                       model='encoder', penalty_amount=encoder_penalty_amount)
    if E_penalty:
        losses.encoder = E_penalty


def generator_routine(data, models, losses, results, viz, measure=None, generator_loss_type='l2',
                      noise_type=None, output_nonlin=None, offset=None):
    Z, Y_P, Y_Q, U, X_P = data.get_batch('z', 'y_p', 'y_q', 'u', 'images')
    Y_P = shape_noise(Y_P, U, noise_type)
    Y_Q = shape_noise(Y_Q, U, noise_type, offset=offset)

    generator = models.generator

    X_Q = generator(Z, nonlinearity=F.tanh)
    _, W_P, _ = encode(models, X_P, None, output_nonlin=output_nonlin, noise_type=noise_type)
    _, W_Q, _ = encode(models, X_Q, None, output_nonlin=output_nonlin, noise_type=noise_type)
    viz.add_image(X_Q, name='generated')

    if generator_loss_type == 'l2':
        generator_loss = (W_P.mean() - W_Q.mean()) ** 2
    else:
        raise NotImplementedError(generator_loss_type)

    losses.generator = generator_loss


# CORTEX ===============================================================================================================
# Must include `BUILD` and `TRAIN_ROUTINES`

def BUILD(data, models, encoder_type='convnet', generator_type='convnet', dim_embedding=1, dim_z=64,
          noise_type=None, generator_noise_type='normal', discriminator_args=dict(), generator_args=dict()):
    x_shape = data.get_dims('x', 'y', 'c')
    data.add_noise('z', dist=generator_noise_type, size=dim_z)
    if noise_type == 'hypercubes':
        noise = 'uniform'
    else:
        noise = 'normal'
    data.add_noise('y_p', dist=noise, size=dim_embedding)
    data.add_noise('y_q', dist=noise, size=dim_embedding)
    data.add_noise('u', dist='uniform', size=1)

    Encoder, encoder_args = update_encoder_args(x_shape, model_type=encoder_type, encoder_args=discriminator_args)
    Decoder, generator_args = update_decoder_args(x_shape, model_type=generator_type, decoder_args=generator_args)

    encoder = Encoder(x_shape, dim_out=dim_embedding, **encoder_args)
    generator = Decoder(x_shape, dim_in=dim_z, **generator_args)

    build_discriminator(models, dim_embedding, key='real_discriminator')
    build_discriminator(models, dim_embedding, key='fake_discriminator')

    models.update(generator=generator, encoder=encoder)


TRAIN_ROUTINES = dict(discriminators=discriminator_routine, encoder=encoder_routine, generator=generator_routine,
                      penalty=penalty_routine)
TEST_ROUTINES = dict(penalty=None)

INFO = dict(measure=dict(choices=['GAN', 'JSD', 'KL', 'RKL', 'X2', 'H2', 'DV', 'W1'],
                         help='GAN measure. {GAN, JSD, KL, RKL (reverse KL), X2 (Chi^2), H2 (squared Hellinger), '
                              'DV (Donsker Varahdan KL), W1 (IPM)}'),
            noise_type=dict(choices=['hypercubes', 'unitball', 'unitsphere'],
                            help='Type of noise to match encoder output to.'),
            dim_z=dict(help='Input dimension to the generator.'),
            dim_embedding=dict(help='Embedding dimension.'),
            offset=dict(help='Offset of Q mode from P mode in discriminator output.'),
            generator_noise_type=dict(help='Type of noise distribution for input to the generator.'),
            output_nonlin=dict(help='Apply nonlinearity at the output of encoder. Will be chosen according to `noise_type`.'),
            generator_loss_type=dict(choices=['l2'], help='Generator loss type.'),
            penalty_amount=dict(help='Amount of gradient penalty for the discriminators.'),
            encoder_penalty_amount=dict(help='Amount of gradient penalty for the encoder.'),
)

DEFAULT_CONFIG = dict(train=dict(save_on_lowest='losses.vae'))