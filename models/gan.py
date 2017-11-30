'''Simple classifier model

'''

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from convnets import SimpleConvEncoder as Discriminator
from conv_decoders import SimpleConvDecoder as Generator


logger = logging.getLogger('cortex.models' + __name__)

GLOBALS = {'DIM_X': None, 'DIM_Y': None, 'DIM_C': None, 'DIM_L': None, 'DIM_Z': None}

discriminator_args_ = dict(dim_h=64, batch_norm=True, nonlinearity='LeakyReLU',
                           f_size=4, stride=2, pad=1, min_dim=4)

generator_args_ = dict(dim_h=64, batch_norm=True, nonlinearity='ReLU',
                       f_size=4, stride=2, pad=1, n_steps=4)

DEFAULTS = dict(
    data=dict(batch_size=128, noise_variables=dict(z=('normal', 64))),
    optimizer=dict(
        optimizer='Adam',
        learning_rate=1e-4,
    ),
    model=dict(),
    procedures=dict(measure='proxy_gan'),
    train=dict(
        epochs=200,
        summary_updates=100,
        archive_every=10
    )
)


def f_divergence(measure, real_out, fake_out, boundary_seek=False):
    if measure in ('gan', 'proxy_gan'):
        r = -F.softplus(-real_out)
        f = F.softplus(-fake_out) + fake_out
        w = torch.exp(fake_out)
        b = fake_out ** 2

    elif measure == 'jsd':
        r = torch.log(2.) - F.softplus(-real_out)
        f = F.softplus(-fake_out) + fake_out + torch.log(2.)
        w = torch.exp(fake_out)
        b = fake_out ** 2

    elif measure == 'xs':
        r = real_out ** 2
        f = -0.5 * ((torch.sqrt(fake_out ** 2) + 1.) ** 2)
        w = 0.5 * (1. - 1. / torch.sqrt(fake_out ** 2))
        b = (fake_out / 2.) ** 2

    elif measure == 'kl':
        r = real_out + 1.
        f = torch.exp(fake_out)
        w = torch.exp(fake_out)
        b = fake_out ** 2

    elif measure == 'rkl':
        r = -torch.exp(-real_out)
        f = fake_out - 1.
        w = torch.exp(fake_out)
        b = fake_out ** 2

    elif measure == 'dv':
        r = real_out
        f = torch.log(torch.exp(fake_out))
        w = torch.exp(fake_out)
        b = fake_out ** 2

    elif measure == 'sh':
        r = 1. - torch.exp(-real_out)
        f = torch.exp(fake_out) - 1.
        w = torch.exp(fake_out)
        b = fake_out ** 2

    else:
        raise NotImplementedError(loss)

    d_loss = torch.mean(f - r)

    if boundary_seek:
        g_loss = torch.mean(b)
    elif measure == 'proxy_gan':
        g_loss = torch.mean(F.softplus(-fake_out))
    else:
        g_loss = -torch.mean(f)

    return d_loss, g_loss, r, f, w, b

def gan(nets, inputs, measure=None, boundary_seek=False):
    discriminator = nets['discriminator']
    generator = nets['generator']
    gen_out = generator(inputs['z'], nonlinearity=F.tanh)

    real_out = discriminator(inputs['images'])
    fake_out = discriminator(gen_out)

    d_loss, g_loss, r, f, w, b = f_divergence(measure, real_out, fake_out, boundary_seek=boundary_seek)
    results = dict(g_loss=g_loss.data[0], d_loss=d_loss.data[0], boundary=torch.mean(b).data[0],
                   real=torch.mean(r).data[0], fake=torch.mean(f).data[0], w=torch.mean(w).data[0])
    samples = dict(generated=0.5*(gen_out.data+1.), real=0.5*(inputs['images'].data+1.))
    return dict(generator=g_loss, discriminator=d_loss), results, samples, 'boundary'


def build_model(loss=None, discriminator_args=None, generator_args=None):
    discriminator_args = discriminator_args or {}
    generator_args = generator_args or {}
    d_args = discriminator_args_
    d_args.update(**discriminator_args)
    g_args = generator_args_
    g_args.update(**generator_args)

    shape = (DIM_X, DIM_Y, DIM_C)

    discriminator = Discriminator(shape, dim_out=1, **d_args)
    generator = Generator(shape, dim_in=64, **g_args)
    logger.debug(discriminator)
    logger.debug(generator)

    return dict(generator=generator, discriminator=discriminator), gan


