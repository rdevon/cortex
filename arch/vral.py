'''Simple GAN model

'''

import logging

import torch
from torch import autograd
from torch.autograd import Variable
import torch.nn.functional as fun

from .gan import apply_penalty, f_divergence
from .modules.densenet import DenseNet


logger = logging.getLogger('cortex.models' + __name__)

resnet_discriminator_args_ = dict(dim_h=64, batch_norm=False, f_size=3, n_steps=3)
resnet_generator_args_ = dict(dim_h=64, batch_norm=True, f_size=3, n_steps=3)

mnist_discriminator_args_ = dict(dim_h=64, batch_norm=True, f_size=5, pad=2, stride=2, min_dim=7,
                                 nonlinearity='LeakyReLU')
mnist_generator_args_ = dict(dim_h=64, batch_norm=True, f_size=4, pad=1, stride=2, n_steps=2)

dcgan_discriminator_args_ = dict(dim_h=64, batch_norm=False, n_steps=3, nonlinearity='ReLU')
dcgan_generator_args_ = dict(dim_h=64, batch_norm=True, n_steps=3)

DEFAULTS = dict(
    data=dict(batch_size=dict(train=64, test=64), skip_last_batch=True,
              noise_variables=dict(z=('normal', 128), r=('normal', 1), f=('normal', 1))),
    optimizer=dict(
        optimizer='Adam',
        learning_rate=1e-4,
        clipping=dict(meta_discriminators=1.0),
        updates_per_model=dict(discriminator=1, generator=1, meta_discriminators=1)
    ),
    model=dict(model_type='dcgan', dim_d=1, dim_e=1, discriminator_args=None, generator_args=None),
    procedures=dict(measure='proxy_gan', boundary_seek=False, penalty=1.0, meta_penalty=0.0),
    train=dict(
        epochs=200,
        summary_updates=100,
        archive_every=10
    )
)

def setup(model=None, data=None, procedure=None, **kwargs):
    data['noise_variables']['r'] = ('normal', model['dim_d'])
    data['noise_variables']['f'] = ('normal', model['dim_d'])


def vral(nets, data_handler, measure=None, boundary_seek=False, penalty=None,
         real_mu=1.0, fake_mu=0.0, lam=100., vral_type=None, meta_penalty=None):
    # Variables
    X = data_handler['images']
    Z = data_handler['z']
    Rr = data_handler['r']
    Fr = data_handler['f']

    # Nets
    discriminator, topnet = nets['discriminator']
    generator = nets['generator']
    real_discriminator, fake_discriminator = nets['meta_discriminators']
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
        assert False, 'Doesnt work right now'
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
        d_loss_r, g_loss_r, _, _, _, _ = f_divergence(measure, real_r, real_f, boundary_seek=boundary_seek)
        d_loss_f, g_loss_f, _, ff, _, _ = f_divergence(measure, fake_r, fake_f, boundary_seek=boundary_seek)
        d_loss = lam * (g_loss_r + g_loss_f) + Ff.mean()

        g_loss = -Ff.mean()
    else:
        d_loss_r, g_loss_r, _, _, _, _ = f_divergence(measure, real_r, real_f, boundary_seek=boundary_seek)
        d_loss_f, g_loss_f, _, _, _, _ = f_divergence(measure, fake_r, fake_f, boundary_seek=boundary_seek)

        d_loss = g_loss_r + g_loss_f
        g_loss = (Ff.mean() - Rf.mean()) ** 2

    results = dict(g_loss=g_loss.data[0], d_loss=d_loss.data[0], rd_loss=d_loss_r.data[0], fd_loss=d_loss_f.data[0])
                   #real=torch.mean(Rf).data[0], fake=torch.mean(Ff).data[0])
    samples = dict(images=dict(generated=0.5 * (gen_out + 1.).data))#, real=0.5 * (X + 1.).data),
                   #histograms=dict(discriminator_output=dict(fake=Ff.view(-1).data, real=Rf.view(-1).data)))

    if meta_penalty:
        assert False
        p_term_r = apply_penalty(data_handler, real_discriminator, Rr, Rf, measure)
        p_term_f = apply_penalty(data_handler, fake_discriminator, Fr, Ff, measure)

        d_loss_r += meta_penalty * p_term_r.mean()
        d_loss_f += meta_penalty * p_term_f.mean()
        results['real gradient penalty'] = p_term_r.mean().data[0]
        results['fake gradient penalty'] = p_term_f.mean().data[0]

    if penalty:
        real = Variable(X.data.cuda(), requires_grad=True)
        fake = Variable(gen_out.data.cuda(), requires_grad=True)
        real_out = topnet(discriminator(real))
        fake_out = topnet(discriminator(fake))

        g_r = autograd.grad(outputs=real_out, inputs=real, grad_outputs=torch.ones(real_out.size()).cuda(),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]
        g_f = autograd.grad(outputs=fake_out, inputs=fake, grad_outputs=torch.ones(fake_out.size()).cuda(),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]

        g_r = (g_r.view(g_r.size()[0], -1) ** 2).sum(1)
        g_f = (g_f.view(g_f.size()[0], -1) ** 2).sum(1)
        g_p = 0.5 * (g_r.mean() + g_f.mean())

        d_loss += penalty * g_p
        #results['gradient penalty'] = g_p.data[0]

    loss = dict(generator=g_loss, meta_discriminators=d_loss_r+d_loss_f, discriminator=d_loss)
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

    return dict(generator=generator, discriminator=[discriminator, topnet], meta_discriminators=[real_discriminator, fake_discriminator]), vral


