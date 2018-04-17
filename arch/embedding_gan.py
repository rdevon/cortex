'''Simple embedding GAN model

'''

import torch
from torch import autograd
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F

from .modules.densenet import DenseNet
from .classifier import classify


resnet_discriminator_args_ = dict(dim_h=64, batch_norm=True, f_size=3, n_steps=4)
resnet_generator_args_ = dict(dim_h=64, batch_norm=True, f_size=3, n_steps=4)

mnist_discriminator_args_ = dict(dim_h=64, batch_norm=True, f_size=5, pad=2, stride=2, min_dim=7,
                                 nonlinearity='LeakyReLU')
mnist_generator_args_ = dict(dim_h=64, batch_norm=True, f_size=4, pad=1, stride=2, n_steps=2)

dcgan_discriminator_args_ = dict(dim_h=64, batch_norm=True, n_steps=3, nonlinearity='LeakyReLU')
dcgan_generator_args_ = dict(dim_h=64, batch_norm=True, n_steps=3)


def discriminator_routine(data, models, losses, results, viz, penalty_amount=0.):
    Z, X_P, A = data.get_batch('z', 'images', 'attributes')
    discriminator = models['discriminator']
    generator = models['generator']

    X_Q = generator(torch.cat([Z, A], 1), nonlinearity=F.tanh)
    W_P = discriminator(X_P)
    W_Q = discriminator(X_Q)

    S_P = (W_P * A).sum(1)
    S_Q = (W_Q * A).sum(1)

    d_loss = S_Q.mean() - S_P.mean()

    if penalty_amount:
        X_P = Variable(X_P.data.cuda(), requires_grad=True)
        X_Q = Variable(X_Q.data.cuda(), requires_grad=True)
        W_P = discriminator(X_P)
        W_Q = discriminator(X_Q)
        S_P = (W_P * A).sum(1)
        S_Q = (W_Q * A).sum(1)

        G_P = autograd.grad(outputs=S_P, inputs=X_P, grad_outputs=torch.ones(S_P.size()).cuda(),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]
        G_Q = autograd.grad(outputs=S_Q, inputs=X_Q, grad_outputs=torch.ones(S_Q.size()).cuda(),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]

        G_P = G_P.view(G_P.size()[0], -1)
        G_Q = G_Q.view(G_Q.size()[0], -1)

        G_P = (G_P ** 2).sum(1)
        G_Q = (G_Q ** 2).sum(1)

        p_term = 0.5 * (G_P.mean() + G_Q.mean())

        d_loss += penalty_amount * p_term
        results['gradient penalty'] = p_term.data[0]

    losses.update(discriminator=d_loss)
    results.update(S_P=S_P.mean().data[0], fake=S_Q.mean().data[0])
    viz.add_image(X_P, name='ground truth')
    viz.add_histogram(dict(fake=S_Q.view(-1).data, real=S_P.view(-1).data), name='discriminator output')


def generator_routine(data, models, losses, results, viz):
    Z, A = data.get_batch('z','attributes')
    discriminator = models['discriminator']
    generator = models['generator']

    X_Q = generator(torch.cat([Z, A], 1), nonlinearity=F.tanh)
    W_Q = discriminator(X_Q)
    S_Q = (W_Q * A).sum(1)

    losses.update(generator=-S_Q.mean())
    viz.add_image(X_Q, name='generated')


def classifier_routine(data, models, losses, results, viz, **kwargs):
    Z, X_P, A = data.get_batch('z', 'images', 'attributes')
    discriminator = models['discriminator']
    generator = models['generator']
    classifier_f, classifier_r = models['classifier']

    X_Q = generator(torch.cat([Z, A], 1), nonlinearity=F.tanh)
    W_P = discriminator(X_P)
    W_Q = discriminator(X_Q)

    classifiers = dict(classifier_fake=classifier_f, classifier_real=classifier_r)

    classify(data, classifiers, losses, results, viz, key='classifier_real', aux_inputs=W_P, **kwargs)
    classify(data, classifiers, losses, results, viz, key='classifier_fake', aux_inputs=W_Q, aux_viz=X_Q, **kwargs)
    losses['classifier'] = losses['classifier_fake'] + losses['classifier_real']


def build_model(data, models, dim_embedding=312, model_type='convnet', discriminator_args=None, generator_args=None):
    discriminator_args = discriminator_args or {}
    generator_args = generator_args or {}
    shape = data.get_dims('x', 'y', 'c')
    dim_a, dim_l = data.get_dims('a', 'labels')
    dim_z = data.get_dims('z')

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

    models.update(generator=generator, discriminator=discriminator, classifier=(classifier_f, classifier_r))


DEFAULT_CONFIG = dict(
    data=dict(batch_size=dict(train=64, test=1000), skip_last_batch=True,
              noise_variables=dict(z=('normal', 64), e=('uniform', 1))),
    optimizer=dict(
        optimizer='Adam',
        learning_rate=1e-4,
        updates_per_model=dict(discriminator=1, generator=1, classifier=1)
    ),
    model=dict(model_type='dcgan', discriminator_args=None, generator_args=None, dim_embedding=312),
    routines=dict(discriminator=dict(penalty_amount=1.0),
                  generator=dict(),
                  classifier=dict(criterion=nn.CrossEntropyLoss())),
    train=dict(
        epochs=1000,
        summary_updates=100,
        archive_every=10
    )
)


ROUTINES = dict(discriminator=discriminator_routine, generator=generator_routine, classifier=classifier_routine)