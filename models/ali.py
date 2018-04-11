'''Adversarially learned inference and Bi-GAN
'''

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
from torch.autograd import Variable

from .gan import f_divergence
from .modules.densenet import DenseNet


logger = logging.getLogger('cortex.models' + __name__)

mnist_encoder_args_ = dict(dim_h=64, batch_norm=True, f_size=5, pad=2, stride=2, min_dim=7, dim_out=1028)
mnist_decoder_args_ = dict(dim_h=64, batch_norm=True, f_size=4, pad=1, stride=2, n_steps=2)
convnet_encoder_args_ = dict(dim_h=64, batch_norm=True, n_steps=3, fully_connected_layers=[1028])
convnet_decoder_args_ = dict(dim_h=64, batch_norm=True, n_steps=3)

DEFAULTS = dict(
    data=dict(batch_size=dict(train=64, test=640),
              noise_variables=dict(z=('normal', 64))),
    optimizer=dict(
        optimizer='Adam',
        learning_rate=1e-4,
        updates_per_model=dict(discriminator=1, nets=1, ss_nets=1)
    ),
    model=dict(model_type='convnet', dim_z=64, encoder_args=None, decoder_args=None, discriminator_args=None),
    procedures=dict(measure='gan', boundary_seek=False, penalty_type='gradient_norm', penalty=1.0),
    train=dict(
        epochs=500,
        summary_updates=100,
        archive_every=10
    )
)


def setup(model=None, data=None, **kwargs):
    data['noise_variables']['z'] = (data['noise_variables']['z'][0], model['dim_z'])


def build_graph(nets, data_handler, measure=None, boundary_seek=False, penalty=None, penalty_type='gradient_norm'):
    X_r, Y, Z_f = data_handler.get_batch('images', 'targets', 'z')

    encoder, decoder = nets['nets']
    classifier, decoder2 = nets['ss_nets']
    x_disc, z_disc, topnet = nets['discriminator']

    X_f = decoder(Z_f, nonlinearity=F.tanh)
    Z_r = encoder(X_r)
    X_c = decoder(Z_r, nonlinearity=F.tanh)

    W_r = x_disc(X_r, nonlinearity=F.relu)
    U_r = z_disc(Z_r, nonlinearity=F.relu)
    S_r = topnet(torch.cat([W_r, U_r], 1))
    W_f = x_disc(X_f, nonlinearity=F.relu)
    U_f = z_disc(Z_f, nonlinearity=F.relu)
    S_f = topnet(torch.cat([W_f, U_f], 1))

    d_loss, g_loss, r, f, w, b = f_divergence(measure, S_r, S_f, boundary_seek=boundary_seek)

    Z_t = Variable(Z_r.data.cuda(), requires_grad=False)
    X_d = decoder2(Z_t, nonlinearity=F.tanh)
    dd_loss = ((X_r - X_d) ** 2).sum(1).sum(1).sum(1).mean()

    # classification
    y_hat = classifier(Z_t, nonlinearity=F.log_softmax)
    c_loss = torch.nn.CrossEntropyLoss()(y_hat, Y)

    predicted = torch.max(y_hat.data, 1)[1]
    correct = 100. * predicted.eq(Y.data).cpu().sum() / Y.size(0)

    results = dict(g_loss=g_loss.data[0], d_loss=d_loss.data[0], boundary=torch.mean(b).data[0],
                   real=torch.mean(r).data[0], fake=torch.mean(f).data[0], w=torch.mean(w).data[0],
                   accuracy=correct)

    if penalty:
        X = Variable(X_r.data.cuda(), requires_grad=True)
        Z = Variable(Z_r.data.cuda(), requires_grad=True)
        W = x_disc(X, nonlinearity=F.relu)
        U = z_disc(Z, nonlinearity=F.relu)
        S = topnet(torch.cat([W, U], 1))

        G = autograd.grad(outputs=S, inputs=[X, Z], grad_outputs=torch.ones(S.size()).cuda(),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]

        G = G.view(G.size()[0], -1)
        G = (G ** 2).sum(1).mean()

        d_loss += penalty * G
        results['gradient penalty'] = G.data[0]

    z_s = Z_r / Z_r.std(0)
    z_m = z_s - z_s.mean(0)
    b, dim_z = z_m.size()
    correlations = (z_m.unsqueeze(2).expand(b, dim_z, dim_z) * z_m.unsqueeze(1).expand(b, dim_z, dim_z)).sum(0) / float(b)

    samples = dict(images=dict(reconstruction=0.5 * (X_c + 1.), original=0.5 * (X_r + 1.),
                               generated=0.5 * (X_f + 1.), reconstruction2=0.5 * (X_d + 1.)),
                   heatmaps=dict(correlations=correlations),
                   latents=dict(latent=Z_r.data),
                   labels=dict(latent=Y.data),
                   histograms=dict(generated=dict(fake=S_f.view(-1).data, real=S_r.view(-1).data)))
    losses = dict(nets=g_loss, discriminator=d_loss, ss_nets=c_loss+dd_loss)
    return losses, results, samples, None


def build_model(data_handler, model_type='convnet', dim_z=64, encoder_args=None, decoder_args=None,
                discriminator_args=None):
    encoder_args = encoder_args or {}
    decoder_args = decoder_args or {}
    shape = data_handler.get_dims('x', 'y', 'c')
    dim_l = data_handler.get_dims('labels')[0]

    if model_type == 'convnet':
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
    discriminator_args_.update(fully_connected_layers=[], batch_norm=False)

    encoder = Encoder(shape, dim_out=dim_z, **encoder_args_)
    decoder = Decoder(shape, dim_in=dim_z, **decoder_args_)
    decoder2 = Decoder(shape, dim_in=dim_z, **decoder_args_)
    x_disc = Encoder(shape, dim_out=256, **discriminator_args_)
    z_disc = DenseNet(dim_z, dim_h=[dim_z], dim_out=dim_z)
    topnet = DenseNet(256 + dim_z, dim_h=[512, 128], dim_out=1, batch_norm=False)

    classifier = DenseNet(dim_z, dim_h=[64, 64], dim_out=dim_l, batch_norm=True, dropout=0.2)

    return dict(discriminator=(x_disc, z_disc, topnet), nets=(encoder, decoder), ss_nets=(classifier, decoder2)), build_graph