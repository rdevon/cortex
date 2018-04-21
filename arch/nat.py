'''`Unsupervised learning by learning to predict noise

'''

import logging

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

from .ali import build_extra_networks, network_routine as ali_network_routine
from .gan import apply_gradient_penalty
from .vae import update_decoder_args, update_encoder_args, build_encoder, build_decoder


def setup(model=None, data=None, procedures=None, test_procedures=None, **kwargs):
    global Id
    Id = torch.Tensor(model['dim_embedding'], model['dim_embedding'])
    nn.init.eye(Id)
    Id = torch.autograd.Variable(Id.cuda(), requires_grad=False)


def make_assignment(P, I_, J_, Z_P, Z_Q):
    batch_size = I_.shape[0]

    Z_Q_e = Z_Q.unsqueeze(2).expand(Z_Q.size(0), Z_Q.size(1), Z_Q.size(0))
    Z_P_e = Z_P.unsqueeze(2).expand(Z_P.size(0), Z_P.size(1), Z_P.size(0)).transpose(0, 2)
    h_loss = 1 - nn.CosineSimilarity(dim=1, eps=1e-6)(Z_Q_e, Z_P_e)

    rows, cols = linear_sum_assignment(h_loss.data.cpu().numpy())
    P_n = np.zeros((batch_size, batch_size)).astype('int8')
    P_n[rows, cols] = 1

    for ii, i in enumerate(I_):
        for jj, j in enumerate(J_):
            P[i, j] = P_n[ii, jj]


def get_embeddings(encoder, X, P, C, I, epsilon=1e-6):
    P = Variable(torch.FloatTensor(P[I, :].astype('float32')), requires_grad=False).cuda()
    C = Variable(torch.FloatTensor(C), requires_grad=False).cuda()

    Z_Q = encoder(X)
    Z_Q = Z_Q / (torch.sqrt((Z_Q ** 2).sum(1, keepdim=True)) + epsilon)
    Z_P = torch.mm(P, C)

    return Z_P, Z_Q


def encoder_train_routine(data, models, losses, results, viz):
    X, I = data.get_batch('images', 'index')
    encoder = models['encoder']
    P, C = models['extras'][:2]
    I_ = I.data.cpu().numpy()
    Z_P, Z_Q = get_embeddings(encoder, X, P, C, I_)
    e_loss = -nn.CosineSimilarity(dim=1, eps=1e-6)(Z_P, Z_Q).mean()

    losses.update(encoder=e_loss)
    apply_gradient_penalty(data, models, losses, results, inputs=X, model='encoder', penalty_amount=1.0)


def assign_train(data, models, losses, results, viz):
    X, I = data.get_batch('images', 'index')
    encoder = models['encoder']
    P, C = models['extras'][:2]

    I_ = I.data.cpu().numpy()
    J_ = P[I_, :].argmax(1)

    Z_P, Z_Q = get_embeddings(encoder, X, P, C, I_)

    make_assignment(P, I_, J_, Z_P, Z_Q)


def encoder_test_routine(data, models, losses, results, viz):
    X, I = data.get_batch('images', 'index')
    encoder = models['encoder']
    P, C = models['extras'][2:]
    I_ = I.data.cpu().numpy()
    Z_P, Z_Q = get_embeddings(encoder, X, P, C, I_)
    e_loss = -nn.CosineSimilarity(dim=1, eps=1e-6)(Z_P, Z_Q).mean()

    losses.update(encoder=e_loss)
    apply_gradient_penalty(data, models, losses, results, inputs=X, model='encoder', penalty_amount=1.0)


def network_routine(data, models, losses, results, viz):
    ali_network_routine(data, models, losses, results, viz, encoder_key='encoder')


def assign_test(data, models, losses, results, viz):
    X, I = data.get_batch('images', 'index')
    encoder = models['encoder']
    P, C = models['extras'][2:]

    I_ = I.data.cpu().numpy()
    J_ = P[I_, :].argmax(1)

    Z_P, Z_Q = get_embeddings(encoder, X, P, C, I_)

    make_assignment(P, I_, J_, Z_P, Z_Q)


def build_model(data, models, model_type='convnet', dim_embedding=None, encoder_args=None, decoder_args=None):
    x_shape = data.get_dims('x', 'y', 'c')
    dim_l = data.get_dims('labels')

    Encoder, encoder_args = update_encoder_args(x_shape, model_type=model_type, encoder_args=encoder_args)
    Decoder, decoder_args = update_decoder_args(x_shape, model_type=model_type, decoder_args=decoder_args)
    build_encoder(models, x_shape, dim_embedding, Encoder, fully_connected_layers=[1028], **encoder_args)
    build_extra_networks(models, x_shape, dim_embedding, dim_l, Decoder, **decoder_args)

    N, M = data.get_dims('n_train', 'n_test')
    C_train = np.random.normal(size=(N, dim_embedding))
    C_train = C_train / np.sqrt((C_train ** 2).sum(1, keepdims=True))
    P_train = np.eye(N, dtype='int8')

    C_test = np.random.normal(size=(N, dim_embedding))
    C_test = C_test / np.sqrt((C_test ** 2).sum(1, keepdims=True))
    P_test = np.eye(N, dtype='int8')

    models.update(extras=(P_train, C_train, P_test, C_test))


ROUTINES = dict(encoder=(encoder_train_routine, encoder_test_routine), nets=network_routine,
                extras=(assign_train, assign_test))

DEFAULT_CONFIG = dict(
    data=dict(batch_size=dict(train=64, test=64), skip_last_batch=True),
    optimizer=dict(optimizer='Adam', learning_rate=1e-4),
    model=dict(model_type='convnet', dim_embedding=62, encoder_args=None),
    routines=dict(),
    train=dict(epochs=500, archive_every=10)
)
