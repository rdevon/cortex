'''Simple GAN model

'''

import logging

import torch
from torch import autograd
import torch.nn.functional as F
from torch.autograd import Variable

from .gan import f_divergence
from .modules.densenet import DenseNet


logger = logging.getLogger('cortex.models' + __name__)

resnet_encoder_args_ = dict(dim_h=64, batch_norm=False, f_size=3, n_steps=4)
mnist_encoder_args_ = dict(dim_h=64, batch_norm=False, f_size=5, pad=2, stride=2, min_dim=7, nonlinearity='LeakyReLU')
convnet_encoder_args_ = dict(dim_h=64, batch_norm=True, n_steps=3, nonlinearity='ReLU')

DEFAULTS = dict(
    data=dict(batch_size=dict(train=64, test=640), skip_last_batch=True,
              noise_variables=dict(r=('uniform', 16))),
    optimizer=dict(
        optimizer='Adam',
        learning_rate=5.e-5,
        updates_per_model=dict(discriminator=5, encoder=1, topnet=5, revnet=5)
    ),
    model=dict(model_type='convnet', dim_e=2, encoder_args=None),
    procedures=dict(measure='proxy_gan', boundary_seek=False, penalty=1.0),
    train=dict(
        epochs=200,
        summary_updates=100,
        archive_every=10
    )
)


def train(nets, data_handler, measure=None, boundary_seek=False, penalty=None):
    # Variables
    X = data_handler['images']
    Y = data_handler['targets']
    Rr = data_handler['r']

    # Nets
    discriminator = nets['discriminator']
    encoder = nets['encoder']
    topnet = nets['topnet']
    revnet = nets['revnet']

    z = encoder(X)
    Rf = topnet(z)
    z_r = revnet(Rf)

    real_r = discriminator(Rr)
    real_f = discriminator(Rf)

    d_loss, e_loss, rr, fr, wr, br = f_divergence(measure, real_r, real_f, boundary_seek=boundary_seek)

    t_loss = e_loss
    r_loss = ((z - z_r) ** 2).sum(1).mean()
    t_loss += r_loss

    results = dict(e_loss=e_loss.data[0], d_loss=d_loss.data[0], t_loss=t_loss.data[0], r_loss=r_loss.data[0],
                   real=torch.mean(Rf).data[0], real_std=torch.std(Rf).data[0])
    samples = dict(scatters=dict(real=(z.data, Y.data)),
                   histograms=dict(encoder=dict(fake=Rf.view(-1).data, real=Rr.view(-1).data)))

    if penalty:
        X_ = Variable(X.data.cuda(), requires_grad=True)
        z_ = encoder(X_)
        #Rf_ = topnet(z_)
        #f_ = discriminator(Rf_)
        g = autograd.grad(outputs=z_, inputs=X_, grad_outputs=torch.ones(z_.size()).cuda(),
                          create_graph=True, retain_graph=True, only_inputs=True)[0]

        g_p = (g ** 2).sum(1).sum(1).sum(1)
        results['gradient penalty'] = g_p.mean().data[0]
        e_loss += penalty * g_p.mean()
        #t_loss += penalty * g_p.mean()
        #d_loss += penalty * g_p.mean()

    loss = dict(encoder=e_loss, discriminator=d_loss, topnet=t_loss, r_loss=r_loss)
    return loss, results, samples, 'boundary'


def build_model(data_handler, model_type='convnet', dim_d=16, dim_e=2, encoder_args=None):
    encoder_args = encoder_args or {}
    shape = data_handler.get_dims('x', 'y', 'c')

    if model_type == 'resnet':
        from .modules.resnets import ResEncoder as Encoder
        encoder_args_ = resnet_encoder_args_
    elif model_type == 'convnet':
        from .modules.convnets import SimpleConvEncoder as Encoder
        encoder_args_ = convnet_encoder_args_
    elif model_type == 'mnist':
        from .modules.convnets import SimpleConvEncoder as Encoder
        encoder_args_ = mnist_encoder_args_
    else:
        raise NotImplementedError(model_type)

    encoder_args_.update(**encoder_args)

    if shape[0] == 64:
        encoder_args_['n_steps'] = 4

    encoder = Encoder(shape, dim_out=dim_e, **encoder_args_)
    topnet = DenseNet(dim_e, dim_h=[64, 64], dim_out=dim_d, batch_norm=False)
    revnet = DenseNet(dim_d, dim_h=[64, 64], dim_out=dim_e, batch_norm=False)
    discriminator = DenseNet(dim_d, dim_h=[256, 64], dim_out=1, nonlinearity='ReLU', batch_norm=False)
    logger.debug(discriminator)
    logger.debug(encoder)

    return dict(discriminator=discriminator, topnet=topnet, encoder=encoder, revnet=revnet), train


