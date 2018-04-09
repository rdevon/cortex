'''Simple classifier model. Credit goes to Samuel Lavoie
'''

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .modules.convnets import SimpleConvEncoder as Encoder
from .modules.densenet import DenseNet
from .modules.conv_decoders import SimpleConvDecoder as Decoder
from .modules.modules import View


logger = logging.getLogger('cortex.models' + __name__)

mnist_encoder_args_ = dict(dim_h=64, batch_norm=True, f_size=5, pad=2, stride=2, min_dim=7, dim_out=1028)
mnist_decoder_args_ = dict(dim_h=64, batch_norm=True, f_size=4, pad=1, stride=2, n_steps=2)
convnet_encoder_args_ = dict(dim_h=64, batch_norm=True, n_steps=3, dim_out=1028)
convnet_decoder_args_ = dict(dim_h=64, batch_norm=True, n_steps=3)

DEFAULTS = dict(
    data=dict(batch_size=dict(train=64, test=640),
              noise_variables=dict(z=('normal', 64))),
    optimizer=dict(
        optimizer='Adam',
        learning_rate=1e-4,
    ),
    model=dict(dim_z=64, encoder_args=None, decoder_args=None),
    procedures=dict(criterion=F.mse_loss, beta_kld=1.),
    train=dict(
        epochs=500,
        summary_updates=100,
        archive_every=10
    )
)


def setup(model=None, data=None, **kwargs):
    data['noise_variables']['z'] = (data['noise_variables']['z'][0], model['dim_z'])


class VAE(nn.Module):
    def __init__(self, encoder, decoder, dim_out=None, dim_z=None):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.mu_net = nn.Linear(dim_out, dim_z)
        self.logvar_net = nn.Linear(dim_out, dim_z)
        self.decoder = decoder
        self.mu = None
        self.logvar = None
        self.latent = None

    def reparametrize(self, mu, std):
        if self.training:
            esp = Variable(std.data.new(std.size()).normal_(), requires_grad=False).cuda()
            return mu + std * esp
        else:
            return mu

    def forward(self, x, nonlinearity=None):
        encoded = self.encoder(x, nonlinearity=F.relu)
        self.mu = self.mu_net(encoded)
        self.std = self.logvar_net(encoded).exp_()
        self.latent = self.reparametrize(self.mu, self.std)
        return self.decoder(self.latent, nonlinearity=nonlinearity)


def build_graph(nets, inputs, criterion=None, beta_kld=1.):
    X, Y, Z = inputs.get_batch('images', 'targets', 'z')

    vae_net = nets['vae']
    classifier = nets['classifier']
    outputs = vae_net(X, nonlinearity=F.tanh)

    r_loss = criterion(outputs, X, size_average=False)
    kl = 0.5 * (vae_net.std ** 2 + vae_net.mu ** 2 - 2. * torch.log(vae_net.std) - 1.).sum()

    #classification
    y_hat = classifier(Variable(vae_net.mu.data.cuda(), requires_grad=False), nonlinearity=F.log_softmax)
    c_loss = torch.nn.CrossEntropyLoss()(y_hat, Y)

    predicted = torch.max(y_hat.data, 1)[1]
    correct = 100. * predicted.eq(Y.data).cpu().sum() / Y.size(0)
    gen = vae_net.decoder(Z, nonlinearity=F.tanh)

    z_s = vae_net.latent / vae_net.latent.std(0)
    z_m = z_s - z_s.mean(0)
    b, dim_z = z_m.size()
    correlations = (z_m.unsqueeze(2).expand(b, dim_z, dim_z) * z_m.unsqueeze(1).expand(b, dim_z, dim_z)).sum(0) / float(b)

    samples = dict(images=dict(reconstruction=0.5 * (outputs + 1.), original=0.5 * (X + 1.),
                               generated=0.5 * (gen + 1.)),
                   heatmaps=dict(correlations=correlations),
                   latents=dict(latent=vae_net.latent.data),
                   labels=dict(latent=Y.data))
    losses = dict(vae=r_loss + beta_kld * kl, classifier=c_loss)
    results = dict(loss=r_loss.data[0], kld=kl.data[0], accuracy=correct)
    return losses, results, samples, 'reconstruction'


def build_model(data_handler, model_type='convnet', dim_z=64, encoder_args=None, decoder_args=None):
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

    encoder = Encoder(shape, **encoder_args_)
    decoder = Decoder(shape, dim_in=dim_z, **decoder_args_)
    net = VAE(encoder, decoder, dim_out=encoder_args_['dim_out'], dim_z=dim_z)

    classifier = DenseNet(dim_z, dim_h=[64, 64], dim_out=dim_l, batch_norm=True, dropout=0.2)

    return dict(vae=net, classifier=classifier), build_graph