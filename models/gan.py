'''Simple GAN model

'''

import logging
import math

import torch
from torch import autograd
from torch.autograd import Variable
import torch.nn.functional as F


logger = logging.getLogger('cortex.models' + __name__)

resnet_discriminator_args_ = dict(dim_h=64, batch_norm=True, f_size=3, n_steps=4)
resnet_generator_args_ = dict(dim_h=64, batch_norm=True, f_size=3, n_steps=4)

mnist_discriminator_args_ = dict(dim_h=64, batch_norm=True, f_size=5, pad=2, stride=2, min_dim=7,
                                 nonlinearity='LeakyReLU')
mnist_generator_args_ = dict(dim_h=64, batch_norm=True, f_size=4, pad=1, stride=2, n_steps=2)

dcgan_discriminator_args_ = dict(dim_h=64, batch_norm=True, n_steps=3, nonlinearity='LeakyReLU', spectral_norm=False)
dcgan_generator_args_ = dict(dim_h=64, batch_norm=True, n_steps=3)

DEFAULTS = dict(
    data=dict(batch_size=dict(train=64, test=1000), skip_last_batch=True,
              noise_variables=dict(z=('normal', 64), e=('uniform', 1))),
    optimizer=dict(
        optimizer='Adam',
        learning_rate=1e-4,
        updates_per_model=dict(discriminator=1, generator=1)
    ),
    model=dict(model_type='dcgan', discriminator_args=None, generator_args=None),
    procedures=dict(measure='proxy_gan', boundary_seek=False, penalty_type='gradient_norm', penalty=1.0),
    train=dict(
        epochs=30,
        summary_updates=100,
        archive_every=10
    )
)


def f_divergence(measure, real_out, fake_out, boundary_seek=False):
    log_2 = math.log(2.)

    if measure in ('gan', 'proxy_gan'):
        r = -F.softplus(-real_out)
        f = F.softplus(-fake_out) + fake_out
        if boundary_seek:
            w = torch.exp(fake_out)
            b = fake_out ** 2 + real_out ** 2
        else:
            w = Variable(torch.Tensor([0.]).float()).cuda()
            b = Variable(torch.Tensor([0.]).float()).cuda()

    elif measure == 'jsd':
        r = log_2 - F.softplus(-real_out)
        f = F.softplus(-fake_out) + fake_out - log_2
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

    elif measure == 'w':
        r = real_out
        f = fake_out
        w = fake_out
        b = Variable(torch.Tensor([0.]).float()).cuda()

    else:
        raise NotImplementedError(measure)
    d_loss = f.mean() - r.mean()

    if boundary_seek:
        g_loss = b.mean()

    elif measure == 'proxy_gan':
        g_loss = torch.mean(F.softplus(-fake_out))

    else:
        g_loss = -torch.mean(f)

    return d_loss, g_loss, r, f, w, b


def apply_penalty(data_handler, discriminator, real, fake, measure, penalty_type='gradient_norm'):
    real = Variable(real.data.cuda(), requires_grad=True)
    fake = Variable(fake.data.cuda(), requires_grad=True)
    real_out = discriminator(real)
    fake_out = discriminator(fake)

    if penalty_type == 'gradient_norm':

        g_r = autograd.grad(outputs=real_out, inputs=real, grad_outputs=torch.ones(real_out.size()).cuda(),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]

        g_f = autograd.grad(outputs=fake_out, inputs=fake, grad_outputs=torch.ones(fake_out.size()).cuda(),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]

        g_r = g_r.view(g_r.size()[0], -1)
        g_f = g_f.view(g_f.size()[0], -1)

        if measure in ('gan', 'proxy_gan', 'jsd'):
            g_r = ((1. - F.sigmoid(real_out)) ** 2 * (g_r ** 2)).sum(1)
            g_f = (F.sigmoid(fake_out) ** 2 * (g_f ** 2)).sum(1)

        else:
            g_r = (g_r ** 2).sum(1)
            g_f = (g_f ** 2).sum(1)

        return 0.5 * (g_r.mean() + g_f.mean())

    elif penalty_type == 'interpolate':
        try:
            epsilon = data_handler['e'].view(-1, 1, 1, 1)
        except:
            raise ValueError('You must initiate a uniform random variable `e` to use interpolation')
        interpolations = Variable(((1. - epsilon) * fake + epsilon * real[:fake.size()[0]]).data.cuda(),
                                  requires_grad=True)

        mid_out = discriminator(interpolations)
        g = autograd.grad(outputs=mid_out, inputs=interpolations, grad_outputs=torch.ones(mid_out.size()).cuda(),
                          create_graph=True, retain_graph=True, only_inputs=True)[0]
        s = (g ** 2).sum(1).sum(1).sum(1)
        return ((torch.sqrt(s) - 1.) ** 2).mean()

    elif penalty_type == 'variance':
        _, _, r, f, _, _ = f_divergence(measure, real_out, fake_out)
        var_real = real_out.var()
        var_fake = fake_out.var()

        return (var_real - 1.) ** 2 + (var_fake - 1.) ** 2


    else:
        raise NotImplementedError(penalty_type)

def gan(nets, data_handler, measure=None, boundary_seek=False, penalty=None, penalty_type='gradient_norm'):
    Z = data_handler['z']
    X = data_handler['images']

    discriminator = nets['discriminator']
    generator = nets['generator']
    gen_out = generator(Z, nonlinearity=F.tanh)

    real_out = discriminator(X)
    fake_out = discriminator(gen_out)

    d_loss, g_loss, r, f, w, b = f_divergence(measure, real_out, fake_out, boundary_seek=boundary_seek)

    results = dict(g_loss=g_loss.data[0], d_loss=d_loss.data[0], boundary=torch.mean(b).data[0],
                   real=torch.mean(r).data[0], fake=torch.mean(f).data[0], w=torch.mean(w).data[0],
                   real_var=torch.var(r).data[0], fake_var=torch.var(f).data[0])
    samples = dict(images=dict(generated=0.5 * (gen_out + 1.).data, real=0.5 * (data_handler['images'] + 1.).data),
                   histograms=dict(generated=dict(fake=fake_out.view(-1).data, real=real_out.view(-1).data)))

    if penalty:
        p_term = apply_penalty(data_handler, discriminator, X, gen_out, measure, penalty_type=penalty_type)

        d_loss += penalty * torch.mean(p_term)
        results['gradient penalty'] = torch.mean(p_term).data[0]

    return dict(generator=g_loss, discriminator=d_loss), results, samples, 'boundary'


def build_model(data_handler, model_type='resnet', discriminator_args=None, generator_args=None):
    discriminator_args = discriminator_args or {}
    generator_args = generator_args or {}
    shape = data_handler.get_dims('x', 'y', 'c')
    dim_z = data_handler.get_dims('z')[0]

    if model_type == 'resnet':
        from .modules.resnets import ResEncoder as Discriminator
        from .modules.resnets import ResDecoder as Generator
        discriminator_args_ = resnet_discriminator_args_
        generator_args_ = resnet_generator_args_
    elif model_type == 'dcgan':
        from .modules.conv_decoders import SimpleConvDecoder as Generator
        from .modules.convnets import SimpleConvEncoder as Discriminator
        discriminator_args_ = dcgan_discriminator_args_
        generator_args_ = dcgan_generator_args_
    elif model_type == 'mnist':
        from .modules.conv_decoders import SimpleConvDecoder as Generator
        from .modules.convnets import SimpleConvEncoder as Discriminator
        discriminator_args_ = mnist_discriminator_args_
        generator_args_ = mnist_generator_args_
    else:
        raise NotImplementedError(model_type)

    discriminator_args_.update(**discriminator_args)
    generator_args_.update(**generator_args)

    if shape[0] == 64:
        discriminator_args_['n_steps'] = 4
        generator_args_['n_steps'] = 4

    discriminator = Discriminator(shape, dim_out=1, **discriminator_args_)
    generator = Generator(shape, dim_in=dim_z, **generator_args_)
    logger.debug(discriminator)
    logger.debug(generator)

    return dict(generator=generator, discriminator=discriminator), gan


