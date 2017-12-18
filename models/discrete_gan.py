'''Discrete GANs

'''

import logging
import math

import torch
from torch import autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from gan import apply_penalty, f_divergence
from conv_decoders import SimpleConvDecoder as Generator
from convnets import SimpleConvEncoder as Discriminator


logger = logging.getLogger('cortex.models' + __name__)

GLOBALS = {'DIM_X': None, 'DIM_Y': None, 'DIM_C': None, 'DIM_L': None, 'DIM_Z': None}

discriminator_args_ = dict(dim_h=64, batch_norm=False, f_size=3, pad=1, stride=2, min_dim=3,
                           nonlinearity='LeakyReLU')
generator_args_ = dict(dim_h=64, batch_norm=True, f_size=5, pad=2, stride=1, n_steps=3)

DEFAULTS = dict(
    data=dict(batch_size=64,
              noise_variables=dict(z=('normal', 64),
                                   r=('uniform', 10 * 28 * 28),
                                   u=('uniform', 28 * 28)),
              test_batch_size=64),
    optimizer=dict(
        optimizer='Adam',
        learning_rate=1e-3,
    ),
    model=dict(discriminator_args=discriminator_args_, generator_args=generator_args_),
    procedures=dict(measure='gan', penalty=5.0, n_samples=10),
    train=dict(
        epochs=200,
        summary_updates=100,
        archive_every=10
    )
)


log_Z = Variable(torch.Tensor([0.]).float(), requires_grad=False).cuda()
#log_M = Variable(torch.log(torch.Tensor([10]).float())).cuda()


def log_sum_exp(x, axis=None, keepdims=False):
    x_max = torch.max(x, axis, keepdim=True)[0]
    y = torch.log(torch.exp(x - x_max).sum(axis, keepdim=True)) + x_max
    return y


def discrete_gan(nets, inputs, measure=None, boundary_seek=False, penalty=None,
                 n_samples=10, reinforce=False, gamma=0.95):
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
    R = R.view(M, -1, DIM_C * DIM_X * DIM_Y)
    U.requires_grad = False

    logit = generator(Z)
    assert logit.size() == X.size(), (logit.size(), X.size())

    g_output = F.sigmoid(logit)
    g_output_ = g_output.view(-1, DIM_C * DIM_X * DIM_Y)
    S = (R <= g_output_).float()
    S = S.view(M, -1, DIM_C, DIM_X, DIM_Y)
    S_ = Variable(S.data.cuda(), volatile=True)

    gen_out = (U <= g_output_).float()
    gen_out = gen_out.view(-1, DIM_C, DIM_X, DIM_Y)

    real_out = discriminator(X)
    fake_out = discriminator(S.view(-1, DIM_C, DIM_X, DIM_Y))
    fake_out_ = discriminator(S_.view(-1, DIM_C, DIM_X, DIM_Y))

    # Get the log probabilities of the samples.
    log_g = -((1. - S) * logit + F.softplus(-logit)).sum(2).sum(2).sum(2)
    d_loss, _, r, f, w, b = f_divergence(
        measure, real_out, fake_out.view(M, B, -1),
        boundary_seek=boundary_seek)
    #w = w.detach()
    #w.requires_grad = False
    #print torch.log(w).mean().data[0], fake_out.mean().data[0], fake_out_.mean().data[0]

    if measure in ('gan', 'jsd', 'rkl', 'kl', 'sh', 'proxy_gan', 'dv'):
        log_w = Variable(fake_out_.data.cuda(), requires_grad=False).view(M, B)
        log_Z_est = log_sum_exp(log_w, axis=0)
        log_w_tilde = log_w - log_Z_est
        w_tilde = torch.exp(log_w_tilde)

    elif measure == 'xs':
        w = (fake_out / 2. + 1.).view(M, B)
        w_tilde = w / w.sum(0)
        log_Z_est = torch.log(torch.mean(w))

    else:
        raise NotImplementedError(measure)

    if reinforce:
        r = (log_w - log_Z)
        assert not r.requires_grad
        g_loss = -(r * log_g).sum(0).mean()
    else:
        assert not w_tilde.requires_grad
        g_loss = -(w_tilde * log_g).sum(0).mean()

    results = dict(g_loss=g_loss.data[0], d_loss=d_loss.data[0], boundary=torch.mean(b).data[0],
                   real=torch.mean(r).data[0], fake=torch.mean(f).data[0],
                   gen_out=g_output.mean().data[0], w_tilde=w_tilde.mean().data[0],
                   real_out=real_out.mean().data[0], fake_out=fake_out.mean().data[0])
    samples = dict(images=dict(generated=gen_out.data,
                               real=X.data))

    if penalty:
        p_term = apply_penalty(inputs, discriminator, generator, measure)

        d_loss += penalty * torch.mean(p_term)
        results['gradient penalty'] = torch.mean(p_term).data[0]

    log_Z *= gamma
    log_Z += (1. - gamma) * log_Z_est.mean()
    results.update(log_Z=log_Z.data[0], log_Z_est=log_Z_est.mean().data[0],
                   log_w=log_w.mean().data[0], log_M=log_M, log_g=log_g.mean().data[0])
    return dict(generator=g_loss, discriminator=d_loss), results, samples, 'boundary'



def build_model(loss=None, discriminator_args=None, generator_args=None):
    discriminator_args = discriminator_args or {}
    generator_args = generator_args or {}

    shape = (DIM_X, DIM_Y, DIM_C)

    discriminator = Discriminator(shape, dim_out=1, **discriminator_args)
    generator = Generator(shape, dim_in=64, **generator_args)
    logger.debug(discriminator)
    logger.debug(generator)

    return dict(generator=generator, discriminator=discriminator), discrete_gan