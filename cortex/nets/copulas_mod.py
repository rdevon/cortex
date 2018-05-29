"""
Simple GAN model
"""

import logging
import torch
from torch import autograd
from torch.autograd import Variable
from torchvision.models import DenseNet

LOGGER = logging.getLogger('cortex.models' + __name__)

DEFAULTS = dict(
    data=dict(batch_size=dict(train=64, test=1028), skip_last_batch=True),
    optimizer=dict(
        optimizer='Adam',
        learning_rate=dict(discriminator=1e-4, nets=1e-4, ss_nets=1e-4),
        updates_per_model=dict(discriminator=5, nets=1, ss_nets=1)),
    model=dict(model_type='convnet', dim_embedding=16, dim_noise=16, encoder_args=None, dim_in=64),
    procedures=dict(measure='proxy_gan', boundary_seek=False, penalty=1., noise='uniform'),
    train=dict(
        epochs=500,
        summary_updates=100,
        archive_every=10
    )
)


def setup(model=None, data=None, procedures=None, **kwargs):
    noise = procedures['noise']
    if noise == 'dirichlet':
        noise = 'uniform'
    data['noise_variables'] = dict(y=(noise, model['dim_noise']), z=(noise, 1), x=('normal', model['dim_in']))


def to_one_hot(y, K):
    y_ = torch.unsqueeze(y, 2)

    one_hot = torch.FloatTensor(y.size(0), y.size(1), K).zero_().cuda()
    one_hot.scatter_(2, y_.data.cuda(), 1)
    return Variable(one_hot)


def make_graph(nets, data_handler, measure=None, boundary_seek=False, penalty=None):

    # Variables
    X, Yr, Zr = data_handler.get_batch('x', 'y', 'z')

    # Nets
    discriminator = nets['discriminator']
    encoder, topnet = nets['nets']

    Yf = encoder(X)
    Zf = topnet(Yf)

    real_f = discriminator(torch.cat([Yf, Zf], 1))
    real_r = discriminator(torch.cat([Yr, Zr], 1))

    d_loss, e_loss, rr, fr, wr, br = f_divergence(measure, real_r, real_f, boundary_seek=boundary_seek)

    log_ratio = -0.5 * ((X ** 2).sum(1) - X.size(1).float() * torch.log(2 * torch.pi)) - torch.log(Zf)

    results = dict(e_loss=e_loss.data[0], d_loss=d_loss.data[0], log_ratio=log_ratio.mean())

    if penalty:
        X = Variable(X.data.cuda(), requires_grad=True)
        Yr = Variable(Yr.data.cuda(), requires_grad=True)
        Zr = Variable(Zr.data.cuda(), requires_grad=True)
        Yf = encoder(X)

        Zf = topnet(Yf)
        real = torch.cat([Yr, Zr], 1)
        fake = torch.cat([Yf, Zf], 1)

        real_out = discriminator(real)
        fake_out = discriminator(fake)

        g_r = autograd.grad(outputs=real_out, inputs=real, grad_outputs=torch.ones(real_out.size()).cuda(),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]

        g_f = autograd.grad(outputs=fake_out, inputs=fake, grad_outputs=torch.ones(fake_out.size()).cuda(),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]

        g_r = g_r.view(g_r.size()[0], -1)
        g_f = g_f.view(g_f.size()[0], -1)

        g_r = (g_r ** 2).sum(1)
        g_f = (g_f ** 2).sum(1)

        p_term = 0.5 * (g_r.mean() + g_f.mean())

        d_loss += penalty * p_term.mean()
        e_loss += penalty * p_term.mean()
        results['real gradient penalty'] = p_term.mean().data[0]

    loss = dict(nets=e_loss, discriminator=d_loss)
    return loss, results, None, 'boundary'


def build_model(data_handler, model_type='convnet', use_topnet=False, dim_noise=16, dim_embedding=16,
                encoder_args=None, decoder_args=None):

    if not use_topnet:
        dim_embedding = dim_noise

    dim_x, dim_y = data_handler.get_dims('x', 'y')

    encoder = DenseNet(dim_x, dim_out=dim_y, dim_h=[128, 128, 128], batch_norm=True)
    topnet = DenseNet(dim_y, dim_h=[64, 32, 16], dim_out=dim_noise, batch_norm=True)
    nets = [encoder, topnet]

    discriminator = DenseNet(dim_y+1, dim_h=[128, 64, 32], dim_out=1, batch_norm=False)

    return dict(discriminator=discriminator, nets=nets), make_graph


