'''Simple GAN model

'''

import logging

import torch
from torch.autograd import Variable
import torch.nn.functional as fun

from .gan import apply_penalty, f_divergence
from .modules.densenet import DenseNet


logger = logging.getLogger('cortex.models' + __name__)

resnet_discriminator_args_ = dict(dim_h=64, batch_norm=True, f_size=3, n_steps=3)
resnet_generator_args_ = dict(dim_h=64, batch_norm=True, f_size=3, n_steps=3)

mnist_discriminator_args_ = dict(dim_h=64, batch_norm=True, f_size=5, pad=2, stride=2, min_dim=7,
                                 nonlinearity='LeakyReLU')
mnist_generator_args_ = dict(dim_h=64, batch_norm=True, f_size=4, pad=1, stride=2, n_steps=2)

dcgan_discriminator_args_ = dict(dim_h=64, batch_norm=True, n_steps=3, nonlinearity='ReLU')
dcgan_generator_args_ = dict(dim_h=64, batch_norm=True, n_steps=3)

DEFAULTS = dict(
    data=dict(batch_size=dict(train=64, test=640), skip_last_batch=True,
              noise_variables=dict(z=('normal', 128), r=('normal', 1), f=('normal', 1))),
    optimizer=dict(
        optimizer='Adam',
        learning_rate=1e-4,#dict(discriminator=1e-4, generator=1e-4, topnet=1e-3, real_discriminator=1e-3, fake_discriminator=1e-3),
        updates_per_model=dict(discriminator=1, generator=1, real_discriminator=1, fake_discriminator=1, topnet=1)
    ),
    model=dict(model_type='dcgan', dim_d=1, dim_e=1, discriminator_args=None, generator_args=None),
    procedures=dict(measure='proxy_gan', boundary_seek=False, penalty_type='gradient_norm', penalty=1.0),
    train=dict(
        epochs=200,
        summary_updates=100,
        archive_every=10
    )
)

def setup(model=None, data=None, procedure=None, **kwargs):
    data['noise_variables']['r'] = ('normal', model['dim_d'])
    data['noise_variables']['f'] = ('normal', model['dim_d'])


def vral(nets, data_handler, measure=None, boundary_seek=False, penalty=None, penalty_type='gradient_norm',
         real_mu=1.0, fake_mu=0.0, lam=100., vral_type=None):
    # Variables
    X = data_handler['images']
    Z = data_handler['z']
    Rr = data_handler['r']
    Fr = data_handler['f']

    # Nets
    discriminator = nets['discriminator']
    generator = nets['generator']
    topnet = nets['topnet']
    real_discriminator = nets['real_discriminator']
    fake_discriminator = nets['fake_discriminator']
    gen_out = generator(Z, nonlinearity=fun.tanh)


    Rfphi = discriminator(X)
    Ffphi = discriminator(gen_out)
    Rf = topnet(Rfphi)
    Ff = topnet(Ffphi)


    if vral_type == 'fisher':
        real_r = real_discriminator(Rr)
        real_f = real_discriminator(Rf)
        fake_r = fake_discriminator(Fr)
        fake_f = fake_discriminator(Ff - Ff.mean())
        real_g = real_discriminator(Ff - real_mu)
    else:
        real_r = real_discriminator(Rr)
        real_f = real_discriminator(Rf - real_mu)
        fake_r = fake_discriminator(Fr)
        fake_f = fake_discriminator(Ff - fake_mu)
        real_g = real_discriminator(Ff - real_mu)

    if measure == 'mmd':
        B, D = real_r.size()
        eye_B = Variable(torch.eye(B).cuda())
        def mmd_mat(x, y, remove_diagonal=False):
            mat = x.unsqueeze(2).expand(B, D, D) - y.unsqueeze(2).expand(B, D, D)
            term = torch.exp(-(mat ** 2).sum(1).sum(1))
            if remove_diagonal:
                term = ((1. - eye_B) * term).sum()
                return term / (B * (B - 1.))
            else:
                return term.mean()

        rrr = mmd_mat(real_r, real_r, remove_diagonal=True)
        rff = mmd_mat(real_f, real_f, remove_diagonal=True)
        rrf = mmd_mat(real_r, real_f)

        frr = mmd_mat(fake_r, fake_r, remove_diagonal=True)
        fff = mmd_mat(fake_f, fake_f, remove_diagonal=True)
        frf = mmd_mat(fake_r, fake_f)

        rgg = mmd_mat(real_g, real_g, remove_diagonal=True)
        rfg = mmd_mat(real_g, real_f)

        d_loss_r = -(rrr - 2. * rrf + rff)
        d_loss_f = -(frr - 2. * frf + fff)
        d_loss = -(d_loss_f + d_loss_r)

        #g_loss = -2. * rfg + rgg
        g_loss = (Ff.mean() - Rf.mean()) ** 2
    elif vral_type == 'fisher':
        d_loss_r, g_loss_r, rr, fr, wr, br = f_divergence(measure, real_r, real_f, boundary_seek=boundary_seek)
        d_loss_f, g_loss_f, rf, ff, wf, bf = f_divergence(measure, fake_r, fake_f, boundary_seek=boundary_seek)
        d_loss = lam * (g_loss_r + g_loss_f) + Ff.mean()
        t_loss = (g_loss_r + g_loss_f) + Ff.mean()

        g_loss = -Ff.mean()
    else:
        d_loss_r, g_loss_r, rr, fr, wr, br = f_divergence(measure, real_r, real_f, boundary_seek=boundary_seek)
        d_loss_f, g_loss_f, rf, ff, wf, bf = f_divergence(measure, fake_r, fake_f, boundary_seek=boundary_seek)

        d_loss = g_loss_r + g_loss_f
        t_loss = d_loss
        g_loss = (Ff.mean() - Rf.mean()) ** 2

    results = dict(g_loss=g_loss, d_loss=d_loss, rd_loss=d_loss_r, fd_loss=d_loss_f, t_loss=t_loss,
                   real=torch.mean(Rf), fake=torch.mean(Ff))
    samples = dict(images=dict(generated=0.5 * (gen_out + 1.), real=0.5 * (X + 1.)),
                   histograms=dict(discriminator_output=dict(fake=Ff.view(-1), real=Rf.view(-1))))

    if penalty:
        p_term_r = apply_penalty(data_handler, real_discriminator, Rr, Rf, measure, penalty_type=penalty_type)
        p_term_f = apply_penalty(data_handler, fake_discriminator, Fr, Ff, measure, penalty_type=penalty_type)

        d_loss_r += penalty * p_term_r.mean()
        d_loss_f += penalty * p_term_f.mean()
        results['real gradient penalty'] = p_term_r.mean()
        results['fake gradient penalty'] = p_term_f.mean()

        p_term_d = apply_penalty(data_handler, real_discriminator, X, gen_out, measure, penalty_type=penalty_type)
        d_loss += penalty * p_term_d.mean()
        results['gradient_penalty'] = p_term_d.mean()

    loss = dict(generator=g_loss, real_discriminator=d_loss_r, fake_discriminator=d_loss_f, discriminator=d_loss,
                topnet=t_loss)
    return loss, results, samples, 'boundary'


def build_model(data_handler, model_type='resnet', dim_h=64, dim_d=1, dim_e=1,
                discriminator_args=None, generator_args=None):
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
    elif shape[0] == 128:
        discriminator_args_['n_steps'] = 5
        generator_args_['n_steps'] = 5

    discriminator = Discriminator(shape, dim_out=dim_h, **discriminator_args_)
    topnet = DenseNet(dim_h, dim_h=[], dim_out=dim_d)
    generator = Generator(shape, dim_in=dim_z, **generator_args_)
    real_discriminator = DenseNet(dim_d, dim_h=[64], dim_out=dim_e, nonlinearity='LeakyReLU', batch_norm=False,
                                  layer_norm=False)
    fake_discriminator = DenseNet(dim_d, dim_h=[64], dim_out=dim_e, nonlinearity='LeakyReLU', batch_norm=False,
                                  layer_norm=False)
    logger.debug(discriminator)
    logger.debug(generator)
    logger.debug(real_discriminator)
    logger.debug(fake_discriminator)

    return dict(generator=generator, discriminator=discriminator, real_discriminator=real_discriminator,
                fake_discriminator=fake_discriminator, topnet=topnet), vral


