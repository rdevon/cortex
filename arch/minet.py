'''MINE feature detection

'''

import logging

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch import autograd
from torch.autograd import Variable

from .gan import f_divergence
from .modules.densenet import DenseNet


logger = logging.getLogger('cortex.models' + __name__)

resnet_encoder_args_ = dict(dim_h=64, batch_norm=True, f_size=3, n_steps=3)
resnet_decoder_args_ = dict(dim_h=64, batch_norm=True, f_size=3, n_steps=3)
mnist_encoder_args_ = dict(dim_h=64, batch_norm=True, f_size=5, pad=2, stride=2, min_dim=7)
mnist_decoder_args_ = dict(dim_h=64, batch_norm=True, f_size=4, pad=1, stride=2, n_steps=2)
convnet_encoder_args_ = dict(dim_h=64, batch_norm=True, n_steps=3)
convnet_decoder_args_ = dict(dim_h=64, batch_norm=True, n_steps=3)

DEFAULTS = dict(
    data=dict(batch_size=dict(train=64, test=1028), skip_last_batch=True, duplicate=2),
    optimizer=dict(
        optimizer='Adam',
        learning_rate=dict(discriminator=1e-4, encoder=1e-4, ss_nets=1e-4),
        updates_per_model=dict(discriminator=1, encoder=1, ss_nets=1)),
    model=dict(model_type='convnet', dim_embedding=16, encoder_args=None),
    procedures=dict(measure='jsd', boundary_seek=False, penalty=1.),
    train=dict(
        epochs=500,
        summary_updates=100,
        archive_every=10
    )
)


Id = None

def setup(model=None, data=None, procedures=None, test_procedures=None, **kwargs):
    global Id
    Id = torch.Tensor(model['dim_embedding'], model['dim_embedding'])
    nn.init.eye(Id)
    Id = torch.autograd.Variable(Id.cuda(), requires_grad=False)


def to_one_hot(y, K):
    y_ = torch.unsqueeze(y, 2)

    one_hot = torch.FloatTensor(y.size(0), y.size(1), K).zero_().cuda()
    one_hot.scatter_(2, y_.data.cuda(), 1)
    return Variable(one_hot)


def make_graph(nets, data_handler, measure=None, boundary_seek=False, penalty=None):

    # Variables
    Xr, Xf, T = data_handler.get_batch('1.images', '2.images', '1.targets')

    # Nets
    discriminator, topnet = nets['discriminator']
    encoder = nets['encoder']
    decoder, classifier = nets['ss_nets']

    Z = encoder(Xr)
    Zt = Variable(Z.detach().data.cuda())

    Wr = discriminator(Xr)
    Sr = topnet(torch.cat([Wr, Z], 1))
    Wf = discriminator(Xf)
    Sf = topnet(torch.cat([Wf, Z], 1))

    d_loss, _, r, f, w, b = f_divergence(measure, Sr, Sf, boundary_seek=boundary_seek)
    e_loss = d_loss

    Th = classifier(Zt, nonlinearity=F.log_softmax)
    c_loss = torch.nn.CrossEntropyLoss()(Th, T)
    predicted = torch.max(Th.data, 1)[1]
    correct = 100. * predicted.eq(T.data).cpu().sum() / T.size(0)

    Xc = decoder(Zt, nonlinearity=F.tanh)
    dd_loss = ((Xr - Xc) ** 2).sum(1).sum(1).sum(1).mean()

    z_s = Z / Z.std(0)
    z_m = z_s - z_s.mean(0)
    b, dim_z = z_m.size()
    correlations = (z_m.unsqueeze(2).expand(b, dim_z, dim_z) * z_m.unsqueeze(1).expand(b, dim_z, dim_z)).sum(0) / float(b)
    correlations -= Id

    results = dict(e_loss=e_loss.data[0], d_loss=d_loss.data[0],
                   c_loss=c_loss.data[0], dd_loss=dd_loss.data[0], accuracy=correct)
    samples = dict(scatters=dict(labels=(Z.data, T.data)),
                   images=dict(reconstruction=0.5 * (Xc.data + 1.), original=0.5 * (Xr.data + 1.), other=0.5 * (Xf.data + 1.)),
                   heatmaps=dict(correlations=correlations))

    if penalty:
        X = Variable(Xr.data.cuda(), requires_grad=True)
        Z = Variable(Z.data.cuda(), requires_grad=True)
        W = discriminator(X)
        S = topnet(torch.cat([W, Z], 1))

        G = autograd.grad(outputs=S, inputs=X, grad_outputs=torch.ones(S.size()).cuda(),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]

        G = G.view(G.size()[0], -1)
        G = (G ** 2).sum(1).mean()

        d_loss += penalty * G
        results['gradient penalty'] = G.data[0]

    loss = dict(encoder=e_loss, discriminator=d_loss, ss_nets=(dd_loss+c_loss))
    return loss, results, samples, 'boundary'


def build_model(data_handler, model_type='convnet', dim_embedding=64, encoder_args=None, decoder_args=None):

    encoder_args = encoder_args or {}
    decoder_args = decoder_args or {}
    shape = data_handler.get_dims('x', 'y', 'c')
    dim_l = data_handler.get_dims('labels')[0]

    if model_type == 'resnet':
        from .modules.resnets import ResEncoder as Encoder
        from .modules.resnets import ResDecoder as Decoder
        encoder_args_ = resnet_encoder_args_
        decoder_args_ = resnet_decoder_args_
    elif model_type == 'convnet':
        from .modules.convnets import SimpleConvEncoder as Encoder
        from .modules.conv_decoders import SimpleConvDecoder as Decoder
        encoder_args_ = convnet_encoder_args_
        decoder_args_ = convnet_decoder_args_
    elif model_type == 'mnist':
        from .modules.convnets import SimpleConvEncoder as Encoder
        from .modules.conv_decoders import SimpleConvDecoder as Decoder
        encoder_args_ = mnist_encoder_args_
        decoder_args_ = mnist_decoder_args_
    else:
        raise NotImplementedError(model_type)

    encoder_args_.update(**encoder_args)
    decoder_args_.update(**decoder_args)

    if shape[0] == 64:
        encoder_args_['n_steps'] = 4
        decoder_args_['n_steps'] = 4
    discriminator_args_ = {}
    discriminator_args_.update(**encoder_args_)
    discriminator_args_['batch_norm'] = False

    encoder = Encoder(shape, dim_out=dim_embedding, fully_connected_layers=[1028], **encoder_args_)
    decoder = Decoder(shape, dim_in=dim_embedding, **decoder_args_)
    discriminator = Encoder(shape, dim_out=256, **discriminator_args_)
    topnet = DenseNet(256 + dim_embedding, dim_h=[512, 128], dim_out=1, batch_norm=False)

    classifier = DenseNet(dim_embedding, dim_h=[64, 64], dim_out=dim_l, batch_norm=True, dropout=0.2)

    return dict(discriminator=(discriminator, topnet), encoder=encoder, ss_nets=(decoder, classifier)), make_graph


