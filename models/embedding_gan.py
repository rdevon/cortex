'''Simple embedding GAN model

'''

import logging
import math

import torch
from torch import autograd
from torch.autograd import Variable
import torch.nn.functional as F

from .modules.densenet import DenseNet


logger = logging.getLogger('cortex.models' + __name__)

resnet_discriminator_args_ = dict(dim_h=64, batch_norm=True, f_size=3, n_steps=4)
resnet_generator_args_ = dict(dim_h=64, batch_norm=True, f_size=3, n_steps=4)

mnist_discriminator_args_ = dict(dim_h=64, batch_norm=True, f_size=5, pad=2, stride=2, min_dim=7,
                                 nonlinearity='LeakyReLU')
mnist_generator_args_ = dict(dim_h=64, batch_norm=True, f_size=4, pad=1, stride=2, n_steps=2)

dcgan_discriminator_args_ = dict(dim_h=64, batch_norm=True, n_steps=3, nonlinearity='LeakyReLU')
dcgan_generator_args_ = dict(dim_h=64, batch_norm=True, n_steps=3)


DEFAULTS = dict(
    data=dict(batch_size=dict(train=64, test=1000), skip_last_batch=True,
              noise_variables=dict(z=('normal', 64), e=('uniform', 1))),
    optimizer=dict(
        optimizer='Adam',
        learning_rate=1e-4,
        updates_per_model=dict(discriminator=1, generator=1, classifier=1)
    ),
    model=dict(model_type='dcgan', discriminator_args=None, generator_args=None, dim_embedding=312),
    procedures=dict(measure='proxy_gan', boundary_seek=False, penalty_type='gradient_norm', penalty=1.0),
    train=dict(
        epochs=1000,
        summary_updates=100,
        archive_every=10
    )
)


def build_graph(nets, data_handler, measure=None, boundary_seek=False, penalty=None, penalty_type='gradient_norm'):
    Z = data_handler['z']
    X_r = data_handler['images']
    A = data_handler['attributes']
    T = data_handler['targets']

    discriminator = nets['discriminator']
    generator = nets['generator']
    classifier_f, classifier_r = nets['classifier']

    X_f = generator(torch.cat([Z, A], 1), nonlinearity=F.tanh)
    W_r = discriminator(X_r)
    W_f = discriminator(X_f)

    S_r = (W_r * A).sum(1)
    S_f = (W_f * A).sum(1)

    d_loss = S_f.mean() - S_r.mean()
    g_loss = -d_loss

    T_pf = classifier_f(W_f, nonlinearity=F.log_softmax)
    c_loss_f = torch.nn.CrossEntropyLoss()(T_pf, T)
    predicted_f = torch.max(T_pf.data, 1)[1]
    correct_f = 100. * predicted_f.eq(T.data).cpu().sum() / T.size(0)

    T_pr = classifier_f(W_r, nonlinearity=F.log_softmax)
    c_loss_r = torch.nn.CrossEntropyLoss()(T_pr, T)
    predicted_r = torch.max(T_pr.data, 1)[1]
    correct_r = 100. * predicted_r.eq(T.data).cpu().sum() / T.size(0)

    results = dict(g_loss=g_loss.data[0], d_loss=d_loss.data[0], accuracy_f=correct_f, accuracy_r=correct_r,
                   real=torch.mean(S_r.mean()).data[0], fake=torch.mean(S_f.mean()).data[0])
    samples = dict(images=dict(generated=0.5 * (X_f + 1.).data, real=0.5 * (X_r + 1.).data),
                   histograms=dict(generated=dict(fake=S_f.view(-1).data, real=S_r.view(-1).data)))

    if penalty:
        X_r = Variable(X_r.data.cuda(), requires_grad=True)
        X_f = Variable(X_f.data.cuda(), requires_grad=True)
        W_r = discriminator(X_r)
        W_f = discriminator(X_f)
        S_r = (W_r * A).sum(1)
        S_f = (W_f * A).sum(1)

        g_r = autograd.grad(outputs=S_r, inputs=X_r, grad_outputs=torch.ones(S_r.size()).cuda(),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]

        g_f = autograd.grad(outputs=S_f, inputs=X_f, grad_outputs=torch.ones(S_f.size()).cuda(),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]

        g_r = g_r.view(g_r.size()[0], -1)
        g_f = g_f.view(g_f.size()[0], -1)

        g_r = (g_r ** 2).sum(1)
        g_f = (g_f ** 2).sum(1)

        p_term = 0.5 * (g_r.mean() + g_f.mean())

        d_loss += penalty * p_term
        results['gradient penalty'] = p_term.data[0]

    return dict(generator=g_loss, discriminator=d_loss, classifier=c_loss_r+c_loss_f), results, samples, None



def build_model(data_handler, dim_embedding=312, model_type='convnet', discriminator_args=None, generator_args=None):
    discriminator_args = discriminator_args or {}
    generator_args = generator_args or {}
    shape = data_handler.get_dims('x', 'y', 'c')
    dim_a, = data_handler.get_dims('a')
    dim_z, = data_handler.get_dims('z')
    dim_l, = data_handler.get_dims('labels')

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

    discriminator = Discriminator(shape, dim_out=dim_embedding, **discriminator_args_)
    generator = Generator(shape, dim_in=dim_z+dim_a, **generator_args_)
    classifier_f = DenseNet(dim_embedding, dim_h=[64, 64], dim_out=dim_l, batch_norm=True, dropout=0.2)
    classifier_r = DenseNet(dim_embedding, dim_h=[64, 64], dim_out=dim_l, batch_norm=True, dropout=0.2)

    logger.debug(discriminator)
    logger.debug(generator)

    return dict(generator=generator, discriminator=discriminator, classifier=(classifier_f, classifier_r)), build_graph