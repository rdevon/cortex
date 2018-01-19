'''Simple classifier model

'''

import logging

import torch
from torch import autograd
from torch.autograd import Variable
import torch.nn.functional as F

from modules.convnets import SimpleConvEncoder as Encoder
from modules.densenet import DenseNet
#from resnets import ResEncoder as Encoder
#from resnets import ResDecoder as Generator


logger = logging.getLogger('cortex.models' + __name__)

GLOBALS = {'DIM_X': None, 'DIM_Y': None, 'DIM_C': None, 'DIM_L': None, 'DIM_Z': None}

mnist_discriminator_args_ = dict(dim_h=128, f_size=5, pad=2, stride=2, min_dim=7, nonlinearity='ReLU')
dcgan_discriminator_args_ = dict(dim_h=128, min_dim=4, nonlinearity='LeakyReLU')


DEFAULTS = dict(
    data=dict(batch_size=dict(train=64, test=1000), noise_variables=dict(g=('normal', 2))),
    optimizer=dict(
        optimizer='Adam',
        learning_rate=1e-4,
    ),
    model=dict(discriminator_args=mnist_discriminator_args_),
    procedures=dict(penalty=False),
    train=dict(
        epochs=200,
        summary_updates=100,
        archive_every=10
    )
)


def setup_data(**data_args):
    pass


def feature_map(nets, data_handler, penalty=None):
    encoder = nets['encoder']
    mine_input = nets['mine_input']
    mine_enc = nets['mine_enc']
    g_net = nets['g_net']
    X = data_handler['images']
    Y = data_handler['targets']
    G = data_handler['g']
    data_handler.next()
    X_i = data_handler['images']
    mine_features = mine_input(X)
    mine_features_i = mine_input(X_i)
    features = encoder(X)
    real_g = g_net(G)
    fake_g = g_net(features)

    bs = mine_features_i.size()[0]
    mine_r = torch.cat([mine_features[:bs], features[:bs]], 1)
    mine_f = torch.cat([mine_features_i, features[:bs]], 1)

    real_out = mine_enc(mine_r)
    fake_out = mine_enc(mine_f)

    r = -F.softplus(-real_out)
    f = F.softplus(-fake_out) + fake_out

    rg = -F.softplus(-real_g)
    fg = F.softplus(-fake_g) + fake_g

    m_loss = f.mean() - r.mean()
    e_loss = f.mean() - r.mean() - 0. * fg.mean()
    g_loss = 0. * (fg.mean() - rg.mean())

    results = {}

    if penalty:
        R = Variable(X.data.cuda(), requires_grad=True)
        feat = encoder(R)
        g_r = autograd.grad(outputs=feat.sum(), inputs=R, grad_outputs=torch.ones(feat.size()).cuda(),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]

        g_p = (g_r ** 2).sum(1).sum(1).sum(1)

        results['e term'] = e_loss.data[0]
        results['gradient penalty'] = torch.mean(g_p).data[0]
        e_loss += penalty * torch.mean(g_p)

    results.update(m_loss=m_loss.data[0], e_loss=e_loss.data[0], g_loss=g_loss.data[0],
                   feat_mean=torch.mean(features).data[0],
                   feat_std=torch.std(features).data[0])

    losses = dict(encoder=e_loss, minet=m_loss, g_loss=g_loss)
    viz = dict(scatters=dict(real=(features.data, Y.data)))
    return losses, results, viz, 'g_loss'


def test(nets, inputs, penalty=None):
    pass


def build_model(data_handler, discriminator_args=None):
    discriminator_args = discriminator_args or {}

    shape = data_handler.get_dims('x', 'y', 'c')

    encoder = Encoder(shape, dim_out=2, **discriminator_args)
    mine_input = Encoder(shape, dim_out=2, **discriminator_args)
    mine_enc = DenseNet(4, dim_h=[100, 10], dim_out=1, batch_norm=True)
    g_net = DenseNet(2, dim_h=64, dim_out=1, nonlinearity='ReLU', batch_norm=True)
    logger.debug(encoder)

    return dict(encoder=encoder, mine_input=mine_input, mine_enc=mine_enc, g_net=g_net), feature_map


