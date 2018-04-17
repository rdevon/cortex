'''`Unsupervised learning by learning to predict noise

'''

import logging
import math

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

from .gan import f_divergence, apply_penalty
from .modules.densenet import DenseNet


logger = logging.getLogger('cortex.arch' + __name__)

resnet_encoder_args_ = dict(dim_h=64, batch_norm=True, f_size=3, n_steps=3)
resnet_decoder_args_ = dict(dim_h=64, batch_norm=True, f_size=3, n_steps=3)
mnist_encoder_args_ = dict(dim_h=64, batch_norm=True, f_size=5, pad=2, stride=2, min_dim=7)
mnist_decoder_args_ = dict(dim_h=64, batch_norm=True, f_size=4, pad=1, stride=2, n_steps=2)
convnet_encoder_args_ = dict(dim_h=64, batch_norm=True, n_steps=3)
convnet_decoder_args_ = dict(dim_h=64, batch_norm=True, n_steps=3)

DEFAULTS = dict(
    data=dict(batch_size=dict(train=64, test=64), skip_last_batch=True, test_on_train=True),
    optimizer=dict(
        optimizer='Adam',
        learning_rate=dict(encoder=1e-4, ss_nets=1e-4),
        updates_per_model=dict(encoder=1, ss_nets=1)),
    model=dict(model_type='convnet', dim_embedding=16, encoder_args=None),
    procedures=dict(),
    train=dict(
        epochs=500,
        summary_updates=100,
        archive_every=10
    )
)


Id = None
C = None
P = None
C_t = None
P_t = None


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


def make_assignment(P, I_, J_, Zr, Zf):
    batch_size = I_.shape[0]

    Z_f_e = Zf.unsqueeze(2).expand(Zf.size(0), Zf.size(1), Zf.size(0))
    Z_r_e = Zr.unsqueeze(2).expand(Zr.size(0), Zr.size(1), Zr.size(0)).transpose(0, 2)
    h_loss = 1 - nn.CosineSimilarity(dim=1, eps=1e-6)(Z_f_e, Z_r_e)

    rows, cols = linear_sum_assignment(h_loss.data.cpu().numpy())
    P_n = np.zeros((batch_size, batch_size)).astype('int8')
    P_n[rows, cols] = 1

    for ii, i in enumerate(I_):
        for jj, j in enumerate(J_):
            P[i, j] = P_n[ii, jj]


def train_routine(nets, data, losses, results, assign=True, test=False):
    X, T, I = data.get_batch('images', 'targets', 'index')
    encoder = nets['encoder']
    decoder, classifier = nets['ss_nets']

    I_ = I.data.cpu().numpy()

    if test:
        J_ = P_t[I_, :].argmax(1)
        P_ = Variable(torch.FloatTensor(P_t[I_, :].astype('float32')), requires_grad=False).cuda()
        C_ = Variable(torch.FloatTensor(C_t), requires_grad=False).cuda()
        P = P_t

    else:
        J_ = P[I_, :].argmax(1)
        P_ = Variable(torch.FloatTensor(P[I_, :].astype('float32')), requires_grad=False).cuda()
        C_ = Variable(torch.FloatTensor(C), requires_grad=False).cuda()

    Zf = encoder(X)
    Zf = Zf / (torch.sqrt((Zf ** 2).sum(1, keepdim=True)) + 1e-6)
    Zr = torch.mm(P_, C_)

    if assign:
        make_assignment(P, I_, J_, Zr, Zf)

    e_loss = -nn.CosineSimilarity(dim=1, eps=1e-6)(Zr, Zf).mean()

    Zt = Variable(Zf.detach().data.cuda())
    Xc = decoder(Zt, nonlinearity=F.tanh)

    Th = classifier(Zt, nonlinearity=F.log_softmax)
    c_loss = torch.nn.CrossEntropyLoss()(Th, T)
    predicted = torch.max(Th.data, 1)[1]
    correct = 100. * predicted.eq(T.data).cpu().sum() / T.size(0)

    dd_loss = ((X - Xc) ** 2).sum(1).sum(1).sum(1).mean()

    losses.add(encoder=e_loss, s_nets=(dd_loss+c_loss))
    results.add(e_loss=e_loss.data[0], c_loss=c_loss.data[0], dd_loss=dd_loss.data[0], accuracy=correct)

    return Zr, Zf, Xc


def test_routine(nets, data, losses, results, viz):
    X, T, I = data.get_batch('images', 'targets', 'index')
    global C_t, P_t

    Zr, Zf, Xc = train_routine(nets, data, losses, results, assign=False)

    z_s = Zf / Zf.std(0)
    z_m = z_s - z_s.mean(0)
    b, dim_z = z_m.size()
    correlations = (z_m.unsqueeze(2).expand(b, dim_z, dim_z)
                    * z_m.unsqueeze(1).expand(b, dim_z, dim_z)).sum(0) / float(b)
    correlations -= Id

    viz.add_scatter(labels=(Zf.data, T.data))
    viz.add_histogram(encoder=dict(fake=Zf.view(-1).data, real=Zr.view(-1).data))
    viz.add_image(reconstruction=0.5 * (Xc.data + 1.), original=0.5 * (X.data + 1.))
    viz.add_heatmap(correlations=correlations)


def build_model(data_handler, model_type='convnet', dim_embedding=16, encoder_args=None, decoder_args=None):

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

    encoder = Encoder(shape, dim_out=dim_embedding, fully_connected_layers=[1028], **encoder_args_)
    decoder = Decoder(shape, dim_in=dim_embedding, **decoder_args_)

    classifier = DenseNet(dim_embedding, dim_h=[64, 64], dim_out=dim_l, batch_norm=True, dropout=0.2)

    global C, P
    N, = data_handler.get_dims('n_train')
    C = np.random.normal(size=(N, dim_embedding))
    C = C / np.sqrt((C ** 2).sum(1, keepdims=True))
    P = np.eye(N, dtype='int8')

    return dict(encoder=encoder, ss_nets=(decoder, classifier))


