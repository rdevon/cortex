'''Simple GAN model

'''

import logging

import torch
import torch.nn.functional as fun

from gan import apply_penalty, f_divergence
from modules.densenet import DenseNet


logger = logging.getLogger('cortex.models' + __name__)

resnet_discriminator_args_ = dict(dim_h=64, batch_norm=True, f_size=3, n_steps=3)
resnet_generator_args_ = dict(dim_h=64, batch_norm=True, f_size=3, n_steps=3)

mnist_discriminator_args_ = dict(dim_h=64, batch_norm=True, f_size=5, pad=2, stride=2, min_dim=7,
                                 nonlinearity='LeakyReLU')
mnist_generator_args_ = dict(dim_h=64, batch_norm=True, f_size=4, pad=1, stride=2, n_steps=2)

dcgan_discriminator_args_ = dict(dim_h=64, batch_norm=True, min_dim=4, nonlinearity='ReLU')
dcgan_generator_args_ = dict(dim_h=64, batch_norm=True, n_steps=3)

DEFAULTS = dict(
    data=dict(batch_size=dict(train=64, test=64), skip_last_batch=True,
              noise_variables=dict(z=('normal', 64), r=('normal', 1), f=('normal', 1))),
    optimizer=dict(
        optimizer='Adam',
        learning_rate=5e-4,
        updates_per_model=dict(discriminator=5, generator=1, real_discriminator=5, fake_discriminator=5)
    ),
    model=dict(model_type='dcgan', dim_d=64, discriminator_args=None, generator_args=None),
    procedures=dict(measure='proxy_gan', boundary_seek=False, penalty_type='gradient_norm', penalty=1.0),
    train=dict(
        epochs=200,
        summary_updates=100,
        archive_every=10
    )
)


def vral(nets, data_handler, measure=None, boundary_seek=False, penalty=None, penalty_type='gradient_norm',
         real_mu=1.0, fake_mu=0.0):
    X = data_handler['images']
    Z = data_handler['z']
    Rr = (data_handler['r'] + real_mu).detach()
    Fr = (data_handler['f'] - fake_mu).detach()

    discriminator = nets['discriminator']
    generator = nets['generator']
    real_discriminator = nets['real_discriminator']
    fake_discriminator = nets['fake_discriminator']
    gen_out = generator(Z, nonlinearity=fun.tanh)

    Rf = discriminator(X)
    Ff = discriminator(gen_out)

    real_r = real_discriminator(Rr.t())
    real_f = real_discriminator(Rf.t())
    fake_r = fake_discriminator(Fr.t())
    fake_f = fake_discriminator(Ff.t())
    real_from_fake = real_discriminator(Ff.t())

    '''
    real_r = real_discriminator(Rr)
    real_f = real_discriminator(Rf)
    fake_r = fake_discriminator(Fr)
    fake_f = fake_discriminator(Ff)
    '''

    d_loss_r, g_loss_r, rr, fr, wr, br = f_divergence(measure, real_r, real_f, boundary_seek=boundary_seek)
    d_loss_f, g_loss_f, rf, ff, wf, bf = f_divergence(measure, fake_r, fake_f, boundary_seek=boundary_seek)
    d_loss = g_loss_r + g_loss_f
    #g_loss = real_from_fake.mean()
    g_loss = (Ff.mean() - real_mu) ** 2

    results = dict(g_loss=g_loss, d_loss=d_loss, boundary=torch.mean(br),
                   real=torch.mean(Rf), fake=torch.mean(Ff), w=torch.mean(wr))
    samples = dict(images=dict(generated=0.5 * (gen_out + 1.), real=0.5 * (X + 1.)),
                   histograms=dict(discriminator_output=dict(fake=Ff.view(-1), real=Rf.view(-1))))

    if penalty:
        p_term_r = apply_penalty(data_handler, real_discriminator, Rr.t(), Rf.t(), measure, penalty_type=penalty_type)
        p_term_f = apply_penalty(data_handler, fake_discriminator, Fr.t(), Ff.t(), measure, penalty_type=penalty_type)

        d_loss_r += penalty * torch.mean(p_term_r)
        d_loss_f += penalty * torch.mean(p_term_f)
        results['real gradient penalty'] = torch.mean(p_term_r)
        results['fake gradient penalty'] = torch.mean(p_term_f)

    loss = dict(generator=g_loss, real_discriminator=d_loss_r, fake_discriminator=d_loss_f, discriminator=d_loss)
    return loss, results, samples, 'boundary'


def build_model(data_handler, model_type='resnet', dim_d=1, discriminator_args=None, generator_args=None):
    discriminator_args = discriminator_args or {}
    generator_args = generator_args or {}
    shape = data_handler.get_dims('x', 'y', 'c')

    if model_type == 'resnet':
        from modules.resnets import ResEncoder as Discriminator
        from modules.resnets import ResDecoder as Generator
        discriminator_args_ = resnet_discriminator_args_
        generator_args_ = resnet_generator_args_
    elif model_type == 'dcgan':
        from modules.conv_decoders import SimpleConvDecoder as Generator
        from modules.convnets import SimpleConvEncoder as Discriminator
        discriminator_args_ = dcgan_discriminator_args_
        generator_args_ = dcgan_generator_args_
    elif model_type == 'mnist':
        from modules.conv_decoders import SimpleConvDecoder as Generator
        from modules.convnets import SimpleConvEncoder as Discriminator
        discriminator_args_ = mnist_discriminator_args_
        generator_args_ = mnist_generator_args_
    else:
        raise NotImplementedError(model_type)

    discriminator_args_.update(**discriminator_args)
    generator_args_.update(**generator_args)

    discriminator = Discriminator(shape, dim_out=dim_d, **discriminator_args_)
    generator = Generator(shape, dim_in=64, **generator_args_)
    real_discriminator = DenseNet(data_handler.batch_size['train'], dim_h=[128, 64, 1], dim_out=1,
                                  nonlinearity='LeakyReLU', batch_norm=False)
    fake_discriminator = DenseNet(data_handler.batch_size['train'], dim_h=[128, 64, 1], dim_out=1,
                                  nonlinearity='LeakyReLU', batch_norm=False)
    #real_discriminator = DenseNet(dim_d, dim_h=64, dim_out=1, nonlinearity='LeakyReLU')
    #fake_discriminator = DenseNet(dim_d, dim_h=64, dim_out=1, nonlinearity='LeakyReLU')
    logger.debug(discriminator)
    logger.debug(generator)
    logger.debug(real_discriminator)
    logger.debug(fake_discriminator)

    return dict(generator=generator, discriminator=discriminator, real_discriminator=real_discriminator,
                fake_discriminator=fake_discriminator), vral


