# flake8: noqa
'''Discrete GANs

'''

import logging
import math

import torch
from torch.autograd import Variable
import torch.nn.functional as F

from .gan import apply_penalty, f_divergence
# from conv_decoders import SimpleConvDecoder as Generator
# from convnets import SimpleConvEncoder as Discriminator

from .modules.conv_decoders import MNISTDeConv as Generator
from .modules.convnets import MNISTConv as Discriminator

logger = logging.getLogger('cortex.models' + __name__)

GLOBALS = {'DIM_X': None, 'DIM_Y': None, 'DIM_C': None, 'DIM_L': None, 'DIM_Z': None}

'''
discriminator_args_ = dict(dim_h=64, batch_norm=False, f_size=3, pad=1, stride=2, min_dim=3,
                           nonlinearity='LeakyReLU')
generator_args_ = dict(dim_h=64, batch_norm=True, f_size=5, pad=2, stride=1, n_steps=3)
'''

# discriminator_args_ = dict(dim_h=64, batch_norm=False, f_size=5, pad=2, stride=2, min_dim=7,
#                           nonlinearity='LeakyReLU')
# generator_args_ = dict(dim_h=64, batch_norm=True, f_size=4, pad=2, stride=1, n_steps=2)

discriminator_args_ = dict(dim_h=64, batch_norm=False)
generator_args_ = dict(dim_h=64, batch_norm=True)

DEFAULTS = dict(
    data=dict(batch_size=64,
              noise_variables=dict(z=('uniform', 64),
                                   r=('uniform', 10 * 28 * 28),
                                   r_t=('uniform', 100 * 28 * 28),
                                   u=('uniform', 28 * 28),
                                   e=('uniform', 1)),
              test_batch_size=100),
    optimizer=dict(
        optimizer='Adam',
        learning_rate=1e-3,
        updates_per_model=dict(discriminator=1, generator=1)
    ),
    model=dict(discriminator_args=discriminator_args_, generator_args=generator_args_),
    procedures=dict(measure='gan', penalty=5.0, n_samples=10, penalty_type='gradient_norm'),
    test_procedures=dict(n_samples=100),
    train=dict(
        epochs=200,
        summary_updates=100,
        archive_every=10
    )
)


log_Z = Variable(torch.Tensor([0.]).float(), requires_grad=False).cuda()
# log_M = Variable(torch.log(torch.Tensor([10]).float())).cuda()


def log_sum_exp(x, axis=None):
    x_max = torch.max(x, axis)[0]
    y = torch.log((torch.exp(x - x_max)).sum(axis)) + x_max
    return y


def setup(data=None, optimizer=None, model=None, procedures=None, test_procedures=None, train=None):
    data['noise_variables']['r'] = ('uniform', procedures['n_samples'] * 28 * 28 * 1)
    data['noise_variables']['r_t'] = ('uniform', test_procedures['n_samples'] * 28 * 28 * 1)


def discrete_gan(nets, inputs, measure=None, penalty=None, n_samples=10, reinforce=False, gamma=0.95,
                 penalty_type='gradient_norm', use_beta=False, test_mode=False, use_sm=False):
    global log_Z
    log_M = math.log(n_samples)
    discriminator = nets['discriminator']
    generator = nets['generator']

    M = n_samples
    X = (inputs['images'] >= 0).float()
    Z = inputs['z']
    R = inputs['r']
    U = inputs['u']
    B = inputs['z'].size()[0]
    log_B = math.log(B)

    if R.size()[1] != DIM_C * n_samples * DIM_X * DIM_Y:
        R = inputs['r_t']
    assert R.size() == (B, DIM_C * n_samples * DIM_X * DIM_Y), (R.size(), (B, DIM_C * n_samples * DIM_X * DIM_Y))

    try:
        R = R.view(M, -1, DIM_C * DIM_X * DIM_Y)
    except BaseException:
        R = R.view(M, -1, DIM_C * DIM_X * DIM_Y)
    U.requires_grad = False

    logit = generator(Z)
    assert logit.size()[1:] == X.size()[1:], (logit.size(), X.size())

    g_output = F.sigmoid(logit)
    g_output_ = g_output.view(-1, DIM_C * DIM_X * DIM_Y)

    S = (R <= g_output_).float()
    S = S.view(M, -1, DIM_C, DIM_X, DIM_Y)
    S_ = Variable(S.data.cuda(), volatile=True)
    S = Variable(S.data.cuda(), requires_grad=False)

    gen_out = (U <= g_output_).float()
    gen_out = gen_out.view(-1, DIM_C, DIM_X, DIM_Y)

    real_out = discriminator(X)

    fake_out = discriminator(S.view(-1, DIM_C, DIM_X, DIM_Y))
    fake_out_ = discriminator(S_.view(-1, DIM_C, DIM_X, DIM_Y))
    log_g = -((1. - S) * logit + F.softplus(-logit)).sum(2).sum(2).sum(2)

    if (measure == 'w' and not test_mode) or use_sm:
        fake_out_sm = discriminator(g_output)
        d_loss, g_loss, r, f, w, b = f_divergence(measure, real_out, fake_out_sm)
    else:
        d_loss, g_loss, r, f, w, b = f_divergence(measure, real_out, fake_out.view(M, B, -1))

    if measure in ('gan', 'jsd', 'rkl', 'kl', 'sh', 'proxy_gan', 'dv') and not use_sm:
        log_w = Variable(fake_out_.data.cuda(), requires_grad=False).view(M, B)
        log_beta = log_sum_exp(log_w.view(M * B, -1) - log_M - log_B, axis=0)
        log_alpha = log_sum_exp(log_w - log_M, axis=0)

        if use_beta:
            log_Z_est = log_beta
            log_w_tilde = log_w - log_Z_est - log_M - log_B
        else:
            log_Z_est = log_alpha
            log_w_tilde = log_w - log_Z_est - log_M
        w_tilde = torch.exp(log_w_tilde)

        alpha = torch.exp(log_alpha)
        beta = torch.exp(log_beta)

    elif measure == 'xs':
        w = (fake_out / 2. + 1.).view(M, B)
        w_tilde = w / w.sum(0)
        log_Z_est = torch.log(torch.mean(w))

    elif measure == 'w' or use_sm:
        log_w = Variable(torch.Tensor([0.]).float()).cuda()
        log_Z_est = Variable(torch.Tensor([0.]).float()).cuda()
        w_tilde = Variable(torch.Tensor([0.]).float()).cuda()

    else:
        raise NotImplementedError(measure)

    if measure != 'w' and not use_sm:
        if reinforce:
            r = (log_w - log_Z)
            assert not r.requires_grad
            g_loss = -(r * log_g).sum(0).mean()
        else:
            w_tilde = Variable(w_tilde.data.cuda(), requires_grad=False)
            assert not w_tilde.requires_grad
            if use_beta:
                g_loss = -((w_tilde * log_g).view(M * B)).sum(0).mean()
            else:
                g_loss = -(w_tilde * log_g).sum(0).mean()

    results = dict(g_loss=g_loss.data[0], distance=-d_loss.data[0], boundary=torch.mean(b).data[0],
                   real=torch.mean(r).data[0], fake=torch.mean(f).data[0],
                   gen_out=g_output.mean().data[0], w_tilde=w_tilde.mean().data[0],
                   real_out=real_out.mean().data[0], fake_out=fake_out.mean().data[0])

    if measure != 'w' and not use_sm:
        results.update(alpha=alpha.mean().data[0], log_alpha=log_alpha.mean().data[0],
                       beta=beta.mean().data[0], log_beta=log_beta.mean().data[0])
        results.update(ess=(1. / (w_tilde ** 2).sum(0)).mean().data[0])

    if test_mode or measure == 'w' or use_sm:
        fake_out_sm = discriminator(Variable(g_output.data.cuda(), volatile=True))
        S_th = Variable((g_output >= 0.5).float().data.cuda(), volatile=True)
        fake_out_sam = Variable(fake_out.data.cuda(), volatile=True)
        fake_out_th = discriminator(S_th)
        dist_th = -f_divergence(measure, real_out, fake_out_th)[0]
        dist_sam = -f_divergence(measure, real_out, fake_out_sam)[0]
        dist_sm = -f_divergence(measure, real_out, fake_out_sm)[0]
        results.update(distance_th=dist_th.data[0], distance_sam=dist_sam.data[0],
                       distance_sm=dist_sm.data[0])

    samples = dict(images=dict(generated=gen_out.data,
                               prob=g_output.data,
                               real=X.data))

    if penalty:
        p_term = apply_penalty(inputs, discriminator, X, g_output,
                               measure, penalty_type=penalty_type)
        d_loss += penalty * p_term
        results['gradient penalty'] = p_term.data[0]

    log_Z *= gamma
    log_Z += (1. - gamma) * log_Z_est.mean()
    results.update(log_Z=log_Z.data[0], log_Z_est=log_Z_est.mean().data[0],
                   log_w=log_w.mean().data[0], log_g=log_g.mean().data[0])
    return dict(generator=g_loss, discriminator=d_loss), results, samples, 'boundary'


def mmd(nets, inputs, sample_method='threshold'):
    generator = nets['generator']

    X = (inputs['images'] >= 0).float()
    Z = inputs['z']
    R = inputs['r']
    U = inputs['u']

    logit = generator(Z)
    g_output = F.sigmoid(logit)
    g_output_ = g_output.view(-1, DIM_C * DIM_X * DIM_Y)
    S = (R <= g_output_).float()
    S = S.view(M, -1, DIM_C, DIM_X, DIM_Y)


def evaluate():
    pass


def build_model(loss=None, discriminator_args=None, generator_args=None):
    discriminator_args = discriminator_args or {}
    generator_args = generator_args or {}

    shape = (DIM_X, DIM_Y, DIM_C)

    discriminator = Discriminator(shape, dim_out=1, **discriminator_args)
    generator = Generator(shape, dim_in=64, **generator_args)
    logger.debug(discriminator)
    logger.debug(generator)

    return dict(generator=generator, discriminator=discriminator), discrete_gan
