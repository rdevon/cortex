'''`Unsupervised learning by learning to predict noise

'''

import numpy as np
from progressbar import Bar, ProgressBar, Percentage, Timer
from scipy.optimize import linear_sum_assignment
import torch
from torch import nn

from .ali import build_extra_networks, network_routine as ali_network_routine
from .gan import apply_gradient_penalty
from .vae import update_decoder_args, update_encoder_args, build_encoder


def make_assignment(P, I_, J_, Z_P, Z_Q, results):
    batch_size = I_.shape[0]

    Z_Q_e = Z_Q.unsqueeze(2).expand(Z_Q.size(0), Z_Q.size(1), Z_Q.size(0))
    Z_P_e = Z_P.unsqueeze(2).expand(Z_P.size(0), Z_P.size(1), Z_P.size(0)).transpose(0, 2)
    h_loss = 1 - nn.CosineSimilarity(dim=1, eps=1e-6)(Z_Q_e, Z_P_e)

    rows, cols = linear_sum_assignment(h_loss.data.cpu().numpy())
    P_n = np.zeros((batch_size, batch_size)).astype('int8')
    P_n[rows, cols] = 1
    n_updates = 0
    for ii, i in enumerate(I_):
        for jj, j in enumerate(J_):
            n_updates += (P[i, j] != P_n[ii, jj]).sum().item() / 2.
            P[i, j] = P_n[ii, jj]

    results.update(n_updates_per_batch=n_updates)


def get_embeddings(encoder, X, P, C, I, epsilon=1e-6):
    P = torch.FloatTensor(P[I, :].astype('float32')).cuda()
    C = torch.FloatTensor(C.astype('float32')).cuda()

    Z_Q = encoder(X)
    Z_Q = Z_Q / Z_Q.norm(dim=1, keepdim=True)
    Z_P = torch.mm(P, C)

    return Z_P, Z_Q


def encoder_train_routine(data, models, losses, results, viz):
    X, I = data.get_batch('images', 'index')
    encoder = models['encoder']
    P, C = models['extras']
    N = data.get_dims('n_train')
    P_ = P[:N]
    I_ = I.data.cpu().numpy()
    Z_P, Z_Q = get_embeddings(encoder, X, P_, C, I_)
    e_loss = -nn.CosineSimilarity(dim=1, eps=1e-6)(Z_P, Z_Q).mean()

    losses.update(encoder=e_loss)
    apply_gradient_penalty(data, models, losses, results, inputs=X, model='encoder', penalty_amount=0.)


def encoder_test_routine(data, models, losses, results, viz):
    X, I = data.get_batch('images', 'index')
    encoder = models['encoder']
    P, C = models['extras']
    N = data.get_dims('n_train')
    P_ = P[N:]
    I_ = I.data.cpu().numpy()
    Z_P, Z_Q = get_embeddings(encoder, X, P_, C, I_)
    e_loss = -nn.CosineSimilarity(dim=1, eps=1e-6)(Z_P, Z_Q).mean()

    losses.update(encoder=e_loss)


def network_routine(data, models, losses, results, viz):
    ali_network_routine(data, models, losses, results, viz, encoder_key='encoder')


def assign_train(data, models, losses, results, viz):
    X, I = data.get_batch('images', 'index')
    encoder = models['encoder']
    P, C = models['extras']
    N = data.get_dims('n_train')
    P = P[:N]

    I_ = I.data.cpu().numpy()
    J_ = P[I_, :].argmax(1)

    Z_P, Z_Q = get_embeddings(encoder, X, P, C, I_)

    make_assignment(P, I_, J_, Z_P, Z_Q, results)


def collect_embeddings(data, models, encoder_key='encoder', test=False):
    encoder = models[encoder_key]
    if isinstance(encoder, (list, tuple)):
        encoder = encoder[0]
    encoder.eval()

    data.reset(test=test, string='Performing assignment... ')

    P, C = models['extras']
    N = data.get_dims('n_train')

    if test:
        P = P[N:]
    else:
        P = P[:N]

    C = torch.FloatTensor(C.astype('float32')).cuda()
    ZPs = []
    ZQs = []
    Is = []
    try:
        while True:
            data.next()
            X, I = data.get_batch('images', 'index')
            Z = encoder(X)
            ZQs.append(Z / Z.norm(dim=1, keepdim=True))

            P_ = torch.FloatTensor(P[I, :].astype('float32')).cuda()
            Z_P = torch.mm(P_, C)
            ZPs.append(Z_P)
            Is.append(I)

    except StopIteration:
        pass

    Z_P = torch.cat(ZPs, dim=0)
    Z_Q = torch.cat(ZQs, dim=0)
    I = torch.cat(Is, dim=0)

    return Z_P, Z_Q, I


def assign(data, models, losses, results, viz, batch_size=64):
    N, M = data.get_dims('n_train', 'n_test')

    P, C = models['extras']
    ZP_train, ZQ_train, I_train = collect_embeddings(data, models)
    ZP_test, ZQ_test, I_test = collect_embeddings(data, models, test=True)

    Z_P = torch.cat([ZP_train, ZP_test], dim=0)
    Z_Q = torch.cat([ZQ_train, ZQ_test], dim=0)
    I = torch.cat([I_train, I_test + N], dim=0).data.cpu().numpy()

    idx = np.arange(N + M)
    np.random.shuffle(idx)
    Z_P = Z_P[idx]
    Z_Q = Z_Q[idx]
    I = I[idx]

    n_batches = (N + M) // batch_size

    results['n_updates_per_batch'] = []
    widgets = ['Performing assignments... ', Timer(), Bar()]
    pbar = ProgressBar(widgets=widgets, maxval=n_batches).start()
    for b in range(n_batches):
        I_ = I[b*batch_size:(b+1)*batch_size]
        Z_P_ = Z_P[b*batch_size:(b+1)*batch_size]
        Z_Q_ = Z_Q[b*batch_size:(b+1)*batch_size]
        J_ = P[I_].argmax(1)
        results_ = {}
        make_assignment(P, I_, J_, Z_P_, Z_Q_, results_)
        results['n_updates_per_batch'].append(results_['n_updates_per_batch'])
        pbar.update(b)
    results['n_updates_per_batch'] = np.mean(results['n_updates_per_batch'])


def assign_test(data, models, losses, results, viz):
    X, I = data.get_batch('images', 'index')
    encoder = models['encoder']
    P, C = models['extras']
    N = data.get_dims('n_train')
    P = P[N:]

    I_ = I.data.cpu().numpy()
    J_ = P[I_, :].argmax(1)

    Z_P, Z_Q = get_embeddings(encoder, X, P, C, I_)

    make_assignment(P, I_, J_, Z_P, Z_Q, results)


# ======================================================================================================================


def build_model(data, models, model_type='convnet', dim_embedding=None, encoder_args=None, decoder_args=None):
    x_shape = data.get_dims('x', 'y', 'c')
    dim_l = data.get_dims('labels')

    Encoder, encoder_args = update_encoder_args(x_shape, model_type=model_type, encoder_args=encoder_args)
    Decoder, decoder_args = update_decoder_args(x_shape, model_type=model_type, decoder_args=decoder_args)
    build_encoder(models, x_shape, dim_embedding, Encoder, fully_connected_layers=[1028], dropout=0.5, **encoder_args)
    build_extra_networks(models, x_shape, dim_embedding, dim_l, Decoder, **decoder_args)

    N, M = data.get_dims('n_train', 'n_test')
    C = np.random.normal(size=(N + M, dim_embedding))
    C = C / np.linalg.norm(C, axis=1, keepdims=True)
    P = np.eye(N + M, dtype='int8')

    models.update(extras=(P, C))


ROUTINES = dict(encoder=(encoder_train_routine, encoder_test_routine), nets=network_routine,
                extras=(assign, None))

DEFAULT_CONFIG = dict(
    data=dict(batch_size=dict(train=64, test=64)),
    optimizer=dict(optimizer='Adam', learning_rate=1e-4, train_for=dict(encoder=1, nets=1, extras=1)),
    model=dict(model_type='convnet', dim_embedding=64, encoder_args=None),
    routines=dict(),
    train=dict(epochs=500, archive_every=10, save_on_best='nets_accuracy')
)
