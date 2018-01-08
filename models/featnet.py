'''Simple classifier model

'''

import logging

import torch
from torch import autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from convnets import SimpleConvEncoder as Discriminator
from densenet import DenseNet
#from resnets import ResEncoder as Discriminator
#from resnets import ResDecoder as Generator


logger = logging.getLogger('cortex.models' + __name__)

GLOBALS = {'DIM_X': None, 'DIM_Y': None, 'DIM_C': None, 'DIM_L': None, 'DIM_Z': None}

#discriminator_args_ = dict(dim_h=64, batch_norm=True, f_size=3, n_steps=3)
#generator_args_ = dict(dim_h=64, batch_norm=True, f_size=3, n_steps=3)

discriminator_args_ = dict(dim_h=64, batch_norm=False)
generator_args_ = dict(dim_h=64, batch_norm=False)


DEFAULTS = dict(
    data=dict(batch_size=64),
    optimizer=dict(
        optimizer='Adam',
        learning_rate=1e-4,
    ),
    model=dict(discriminator_args=discriminator_args_, generator_args=generator_args_),
    procedures=dict(penalty=1.0),
    train=dict(
        epochs=200,
        summary_updates=100,
        archive_every=10
    )
)


def feature_map(nets, inputs, penalty=None):
    discriminator = nets['discriminator']
    generator = nets['generator']
    topnet = nets['topnet']
    real_feat = discriminator(inputs['images'])
    fake_feat = generator(inputs['images'])

    #targets = topnet(real_out, F.softmax)
    #targets_ = Variable(targets.data.cuda())
    #pred = topnet(fake_out)
    #pred_ = Variable(pred.data.cuda())

    real_out = topnet(real_feat)
    fake_out = topnet(fake_feat)

    r = -F.softplus(-real_out)
    f = F.softplus(-fake_out) + fake_out
    t_loss = torch.mean(f - r)
    d_loss = torch.mean(-r)
    g_loss = torch.mean(F.softplus(-fake_out))
    #g_loss = torch.mean(-f)

    #g_loss = torch.mean(((1. - targets_) * pred + F.softplus(-pred)).sum(1))
    #d_loss = -g_loss
    #d_loss = torch.mean((targets * pred_).sum(1))

    results = {}

    if penalty:
        real = Variable(inputs['images'].data.cuda(), requires_grad=True)
        real_feat_ = discriminator(real)
        #real_feat_ = Variable(real_feat.data.cuda(), requires_grad=True)
        #fake_feat_ = Variable(fake_feat.data.cuda(), requires_grad=True)
        real_out_ = topnet(real_feat_)
        #fake_out_ = topnet(fake_feat_)

        g_r = autograd.grad(outputs=real_out_, inputs=real, grad_outputs=torch.ones(real_out_.size()).cuda(),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]

        #g_f = autograd.grad(outputs=fake_out_, inputs=fake_feat_, grad_outputs=torch.ones(fake_out_.size()).cuda(),
        #                    create_graph=True, retain_graph=True, only_inputs=True)[0]

        g_p = (g_r ** 2).sum(1).sum(1).sum(1)
        #g_p = (0.5 * ((1. - F.sigmoid(real_out)) ** 2 * (g_r ** 2).sum(1)
        #                  + F.sigmoid(fake_out) ** 2 * (g_f ** 2).sum(1)))

        results['d term'] = d_loss.data[0]
        results['gradient penalty'] = torch.mean(g_p).data[0]
        d_loss += penalty * torch.mean(g_p)
        t_loss += penalty * torch.mean(g_p)

    '''
    results.update(g_loss=g_loss.data[0], d_loss=d_loss.data[0],
                   feat_mean=torch.mean(fake_out).data[0],
                   feat_std=torch.std(fake_out).data[0],
                   norm_term=torch.mean(F.softplus(-pred_).sum(1)).data[0],
                   pred=torch.mean(pred).data[0])
    '''
    results.update(g_loss=g_loss.data[0], d_loss=d_loss.data[0],
                   feat_mean=torch.mean(fake_feat).data[0],
                   feat_std=torch.std(fake_feat).data[0])

    losses = dict(generator=g_loss, discriminator=d_loss, topnet=t_loss)
    viz = dict(scatters=dict(real=(real_feat.data, inputs['targets'].data),
                             fake=(fake_feat.data, inputs['targets'].data)))
    return losses, results, viz, 'g_loss'


def build_model(loss=None, discriminator_args=None, generator_args=None):
    discriminator_args = discriminator_args or {}
    generator_args = generator_args or {}

    shape = (DIM_X, DIM_Y, DIM_C)

    discriminator = Discriminator(shape, dim_out=2, **discriminator_args)
    generator = Discriminator(shape, dim_out=2, **generator_args)
    topnet = DenseNet(2, dim_h=10, dim_out=1, batch_norm=False)
    logger.debug(discriminator)
    logger.debug(generator)

    return dict(generator=generator, discriminator=discriminator, topnet=topnet), feature_map


