'''Implicit feature network

'''

import logging
import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch import autograd
from torch.autograd import Variable

from .gan import f_divergence, apply_penalty
from .modules.densenet import DenseNet


logger = logging.getLogger('cortex.models' + __name__)

resnet_encoder_args_ = dict(dim_h=64, batch_norm=True, f_size=3, n_steps=3)
resnet_decoder_args_ = dict(dim_h=64, batch_norm=True, f_size=3, n_steps=3)
mnist_encoder_args_ = dict(dim_h=64, batch_norm=True, f_size=5, pad=2, stride=2, min_dim=7)
mnist_decoder_args_ = dict(dim_h=64, batch_norm=True, f_size=4, pad=1, stride=2, n_steps=2)
convnet_encoder_args_ = dict(dim_h=64, batch_norm=True, n_steps=3)
convnet_decoder_args_ = dict(dim_h=64, batch_norm=True, n_steps=3)

DEFAULTS = dict(
    data=dict(batch_size=dict(train=64, test=1028), skip_last_batch=True),
    optimizer=dict(
        optimizer='Adam',
        learning_rate=dict(discriminator=1e-4, nets=1e-4, ss_nets=1e-4),
        updates_per_model=dict(discriminator=1, nets=1, ss_nets=1)),
    model=dict(model_type='convnet', dim_embedding=16, dim_noise=16, encoder_args=None, use_topnet=False),
    procedures=dict(measure='jsd', boundary_seek=False, penalty=1., noise_type='hypercubes', noise='uniform'),
    train=dict(
        epochs=500,
        summary_updates=100,
        archive_every=10
    )
)


Id = None

def setup(model=None, data=None, procedures=None, test_procedures=None, **kwargs):
    noise = procedures['noise']
    noise_type = procedures['noise_type']
    if noise_type in ('unitsphere', 'unitball'):
        noise = 'normal'
    data['noise_variables'] = dict(y=(noise, model['dim_noise']))
    data['noise_variables']['u'] = ('uniform', 1)
    procedures['use_topnet'] = model['use_topnet']
    test_procedures['use_topnet'] = model['use_topnet']

    global Id
    Id = torch.Tensor(model['dim_noise'], model['dim_noise'])
    nn.init.eye(Id)
    Id = torch.autograd.Variable(Id.cuda(), requires_grad=False)


def to_one_hot(y, K):
    y_ = torch.unsqueeze(y, 2)

    one_hot = torch.FloatTensor(y.size(0), y.size(1), K).zero_().cuda()
    one_hot.scatter_(2, y_.data.cuda(), 1)
    return Variable(one_hot)


def make_graph(nets, data_handler, measure=None, boundary_seek=False, penalty=None, epsilon=1e-6,
               noise_type='hypercubes', dim_subspace=16, use_topnet=False, output_nonlin=False, noise=None):

    # Variables
    X, T, Yr, U = data_handler.get_batch('images', 'targets', 'y', 'u')
    b = np.zeros(Yr.size(1))

    if noise_type == 'hypercubes':
        b[0] = 1
        a = (np.arange(64) / 64.)[:, None] * b[None, :]

        if dim_subspace and dim_subspace != Yr.size(1):
            assert False
            Y_ = to_one_hot(Variable(
                torch.multinomial(torch.zeros(Yr.size()) + 1. / float(Yr.size(1)), dim_subspace).cuda()
            ), Yr.size(1)).sum(1)
            Yr = Yr * Y_
    elif noise_type == 'sparse':
        b[0] = 1
        a = (np.arange(64) / 64.)[:, None] * b[None, :]
        S = (S < 0.1).float()
        R = V * S
    elif noise_type == 'unitsphere':
        b[0] = 1
        a = ((np.arange(64) / 64.))[:, None] * b[None, :]
        Yr = Yr / (torch.sqrt((Yr ** 2).sum(1, keepdim=True)) + epsilon)
    elif noise_type == 'unitball':
        b[0] = 1
        a = ((np.arange(64) / 64.))[:, None] * b[None, :]
        Yr = Yr / (torch.sqrt((Yr ** 2).sum(1, keepdim=True)) + epsilon) * U.expand(Yr.size())
    else:
        raise ValueError

    # Nets
    discriminator = nets['discriminator']
    if use_topnet:
        encoder, topnet, revnet = nets['nets']
    else:
        encoder, = nets['nets']
    decoder, classifier = nets['ss_nets']

    Zf = encoder(X)
    if output_nonlin:
        if noise_type == 'hypercubes':
            Zf = F.sigmoid(Zf)
        elif noise_type == 'unitsphere':
            Zf = Zf / (torch.sqrt((Zf ** 2).sum(1, keepdim=True)) + epsilon)
        elif noise_type == 'unitball':
            Zf = F.tanh(Zf)

    Zt = Variable(Zf.detach().data.cuda())
    Xc = decoder(Zt, nonlinearity=F.tanh)
    Th = classifier(Zt, nonlinearity=F.log_softmax)

    if use_topnet:
        Yf = topnet(Zf)
        Zr = revnet(Yr)
        real_f = discriminator(torch.cat([Yf, Zf], 1))
        real_r = discriminator(torch.cat([Yr, Zr], 1))
        d_loss, e_loss, rr, fr, wr, br = f_divergence(measure, real_r, real_f, boundary_seek=boundary_seek)
    else:
        Yf = Zf
        real_f = discriminator(Yf)
        real_r = discriminator(Yr)

        d_loss, e_loss, rr, fr, wr, br = f_divergence(measure, real_f, real_r, boundary_seek=boundary_seek)
        #d_loss, e_loss, rr, fr, wr, br = f_divergence(measure, real_r, real_f, boundary_seek=boundary_seek)
    #e_loss *= -1.
    c_loss = torch.nn.CrossEntropyLoss()(Th, T)

    predicted = torch.max(Th.data, 1)[1]
    correct = 100. * predicted.eq(T.data).cpu().sum() / T.size(0)

    dd_loss = ((X - Xc) ** 2).sum(1).sum(1).sum(1).mean()

    I = torch.autograd.Variable(torch.FloatTensor(a).cuda())

    if use_topnet:
        Zi = revnet(I)
        Zs = revnet(Yr)
    else:
        Zi = I
        Zs = Yr

    Xi = decoder(Zi, nonlinearity=F.tanh)
    Xs = decoder(Zs, nonlinearity=F.tanh)

    z_s = Yf / Yf.std(0)
    z_m = z_s - z_s.mean(0)
    b, dim_z = z_m.size()
    correlations = (z_m.unsqueeze(2).expand(b, dim_z, dim_z) * z_m.unsqueeze(1).expand(b, dim_z, dim_z)).sum(0) / float(b)
    correlations -= Id

    results = dict(e_loss=e_loss.data[0], d_loss=d_loss.data[0],
                   c_loss=c_loss.data[0], dd_loss=dd_loss.data[0], accuracy=correct)
    samples = dict(scatters=dict(labels=(Yf.data, T.data)),
                   histograms=dict(encoder=dict(fake=Yf.view(-1).data,
                                                real=Yr.view(-1).data)),
                   images=dict(reconstruction=0.5 * (Xc.data + 1.), original=0.5 * (X.data + 1.),
                               samples=0.5 * (Xs.data + 1.), interpolation=0.5 * (Xi.data + 1.)),
                   heatmaps=dict(correlations=correlations))

    if penalty:
        X = Variable(X.data.cuda(), requires_grad=True)
        #Yr = Variable(Yr.data.cuda(), requires_grad=True)
        Zf = encoder(X)

        if use_topnet:
            Yf = topnet(Zf)
            #Zr = revnet(Yr)
            #real = torch.cat([Yr, Zr], 1)
            fake = torch.cat([Yf, Zf], 1)
        else:
            #real = Yr
            fake = Zf

        #real_out = discriminator(real)
        fake_out = discriminator(fake)

        #g_r = autograd.grad(outputs=real_out, inputs=real, grad_outputs=torch.ones(real_out.size()).cuda(),
        #                    create_graph=True, retain_graph=True, only_inputs=True)[0]

        g_f = autograd.grad(outputs=fake_out, inputs=fake, grad_outputs=torch.ones(fake_out.size()).cuda(),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]

        #g_r = g_r.view(g_r.size()[0], -1)
        g_f = g_f.view(g_f.size()[0], -1)

        #g_r = (g_r ** 2).sum(1)
        g_f = (g_f ** 2).sum(1)

        #p_term = 0.5 * (g_r.mean() + g_f.mean())
        p_term = g_f.mean()

        d_loss += penalty * p_term.mean()
        #e_loss += penalty * p_term.mean()
        results['real gradient penalty'] = p_term.mean().data[0]

    loss = dict(nets=e_loss, discriminator=d_loss, ss_nets=(dd_loss+c_loss))
    return loss, results, samples, 'boundary'


def build_model(data_handler, model_type='convnet', use_topnet=False, dim_noise=16, dim_embedding=16,
                encoder_args=None, decoder_args=None):

    if not use_topnet:
        dim_embedding = dim_noise

    encoder_args = encoder_args or {}
    decoder_args = decoder_args or {}
    shape = data_handler.get_dims('x', 'y', 'c')
    dim_l = data_handler.get_dims('labels')[0]

    if model_type == 'resnet':
        from .modules.resnets import ResEncoder as Encoder
        from .modules.resnets import ResDecoder as Decoder
        encoder_args_ = resnet_encoder_args_
        decoder_args_ = resnet_decoder_args_
        dim_h = [512, 256, 128]
    elif model_type == 'convnet':
        from .modules.convnets import SimpleConvEncoder as Encoder
        from .modules.conv_decoders import SimpleConvDecoder as Decoder
        encoder_args_ = convnet_encoder_args_
        decoder_args_ = convnet_decoder_args_
        dim_h = [2048, 1028, 512]
        dim_top = [128, 128]
    elif model_type == 'mnist':
        from .modules.convnets import SimpleConvEncoder as Encoder
        from .modules.conv_decoders import SimpleConvDecoder as Decoder
        encoder_args_ = mnist_encoder_args_
        decoder_args_ = mnist_decoder_args_
        dim_h = [256, 512, 256]
    else:
        raise NotImplementedError(model_type)

    encoder_args_.update(**encoder_args)
    decoder_args_.update(**decoder_args)

    if shape[0] == 64:
        encoder_args_['n_steps'] = 4
        decoder_args_['n_steps'] = 4

    encoder = Encoder(shape, dim_out=dim_embedding, fully_connected_layers=[1028], **encoder_args_)
    decoder = Decoder(shape, dim_in=dim_embedding, **decoder_args_)

    if use_topnet:
        topnet = DenseNet(dim_embedding, dim_h=dim_top[::-1], dim_out=dim_noise, batch_norm=True)
        revnet = DenseNet(dim_noise, dim_h=dim_top, dim_out=dim_embedding, batch_norm=True)
        nets = [encoder, topnet, revnet]
        dim_d = dim_noise + dim_embedding
    else:
        nets = [encoder]
        dim_d = dim_noise

    discriminator = DenseNet(dim_d, dim_h=dim_h, dim_out=1, batch_norm=False)

    classifier = DenseNet(dim_embedding, dim_h=[64, 64], dim_out=dim_l, batch_norm=True, dropout=0.2)

    return dict(discriminator=discriminator, nets=nets, ss_nets=[decoder, classifier]), make_graph


