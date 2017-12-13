'''Simple classifier model

'''

import logging

import torch
from torch import autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from resnets import ResEncoder as Discriminator
from resnets import ResDecoder as Generator


logger = logging.getLogger('cortex.models' + __name__)

GLOBALS = {'DIM_X': None, 'DIM_Y': None, 'DIM_C': None, 'DIM_L': None, 'DIM_Z': None}

discriminator_args_ = dict(dim_h=64, batch_norm=True, f_size=3, n_steps=3)
generator_args_ = dict(dim_h=64, batch_norm=True, f_size=3, n_steps=3)

DEFAULTS = dict(
    data=dict(batch_size=64, noise_variables=dict(z=('normal', 64))),
    optimizer=dict(
        optimizer='Adam',
        learning_rate=1e-4,
    ),
    model=dict(discriminator_args=discriminator_args_, generator_args=generator_args_),
    procedures=dict(measure='gan', penalty=1.0, boundary_seek=True),
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
        raise NotImplementedError(measure)

    d_loss = torch.mean(f - r)

    if boundary_seek:
        g_loss = torch.mean(b)
    elif measure == 'proxy_gan':
        g_loss = torch.mean(F.softplus(-fake_out))
    else:
        g_loss = -torch.mean(f)

    return d_loss, g_loss, r, f, w, b


def gan(nets, inputs, measure=None, boundary_seek=False, penalty=None):
    discriminator = nets['discriminator']
    generator = nets['generator']
    gen_out = generator(inputs['z'], nonlinearity=F.tanh)

    real_out = discriminator(inputs['images'])
    fake_out = discriminator(gen_out)

    d_loss, g_loss, r, f, w, b = f_divergence(measure, real_out, fake_out, boundary_seek=boundary_seek)

    results = dict(g_loss=g_loss.data[0], d_loss=d_loss.data[0], boundary=torch.mean(b).data[0],
                   real=torch.mean(r).data[0], fake=torch.mean(f).data[0], w=torch.mean(w).data[0])
    samples = dict(generated=0.5 * (gen_out.data + 1.), real=0.5 * (inputs['images'].data + 1.))

    if penalty:
        real = Variable(inputs['images'].data.cuda(), requires_grad=True)
        fake = Variable(gen_out.data.cuda(), requires_grad=True)
        real_out = discriminator(real)
        fake_out = discriminator(fake)

        g_r = autograd.grad(outputs=real_out, inputs=real, grad_outputs=torch.ones(real_out.size()).cuda(),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]

        g_f = autograd.grad(outputs=fake_out, inputs=fake, grad_outputs=torch.ones(fake_out.size()).cuda(),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]

        if measure in ('gan', 'proxy_gan', 'jsd'):
            g_p = (0.5 * ((1. - F.sigmoid(real_out)) ** 2 * (g_r ** 2).sum(1).sum(1).sum(1)
                          + F.sigmoid(fake_out) ** 2 * (g_f ** 2).sum(1).sum(1).sum(1)))

        else:
            g_p = (0.5 * ((g_r ** 2).sum(1).sum(1).sum(1)
                          + (g_f ** 2).sum(1).sum(1).sum(1)))

        g_p = torch.mean(g_p)

        d_loss += penalty * torch.mean(g_p)
        results['gradient penalty'] = torch.mean(g_p).data[0]

    return dict(generator=g_loss, discriminator=d_loss), results, samples, 'boundary'


def build_model(loss=None, discriminator_args=None, generator_args=None):
    discriminator_args = discriminator_args or {}
    generator_args = generator_args or {}

    shape = (DIM_X, DIM_Y, DIM_C)

    discriminator = Discriminator(shape, dim_out=1, **discriminator_args)
    generator = Generator(shape, dim_in=64, **generator_args)
    logger.debug(discriminator)
    logger.debug(generator)

    return dict(generator=generator, discriminator=discriminator), gan


