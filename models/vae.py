'''Simple classifier model. Credit goes to Samuel Lavoie
'''

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from modules.convnets import SimpleConvEncoder as Encoder
from modules.conv_decoders import SimpleConvDecoder as Decoder
from modules.modules import View


logger = logging.getLogger('cortex.models' + __name__)

encoder_args_ = dict(dim_h=64, batch_norm=True, n_steps=3, nonlinearity='ReLU')
decoder_args_ = dict(dim_h=64, batch_norm=True, n_steps=3, dim_in=64)
vae_args_ = dict(dim_h=64, n_steps=3, dim_latent=64)

DEFAULTS = dict(
    data=dict(batch_size=dict(train=64, test=640)),
    optimizer=dict(
        optimizer='Adam',
        learning_rate=1e-4,
    ),
    model=dict(encoder_args=encoder_args_, decoder_args=decoder_args_, vae_args=vae_args_),
    procedures=dict(criterion=F.mse_loss, beta_kld=1.),
    train=dict(
        epochs=200,
        summary_updates=100,
        archive_every=10
    )
)


class Vae(nn.Module):
    def __init__(self, encoder, decoder, dim_h, n_steps, dim_latent, nonlinearity):
        super(Vae, self).__init__()
        e_size = 4 * 4 * dim_h * 2 ** (n_steps - 1)
        self.encoder = nn.Sequential(encoder, View(-1, e_size))
        self.mu_net = nn.Linear(e_size, dim_latent)
        self.logvar_net = nn.Linear(e_size, dim_latent)
        self.decoder = nn.Sequential(decoder, nonlinearity)
        self.mu = self.logvar = self.latent = None

    def reparametrize(self, mu, std):
        if self.training:
            esp = Variable(std.data.new(std.size()).normal_(), requires_grad=False).cuda()
            return esp.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x):
        encoded = self.encoder(x)
        self.mu = self.mu_net(encoded)
        self.std = self.logvar_net(encoded).exp_()
        self.latent = self.reparametrize(self.mu, self.std)
        return self.decoder(self.latent)


def build_graph(net, inputs, criterion=None, beta_kld=1.):
    images = 0.5 * (inputs['images'] + 1.)
    vae_net = net['vae']
    outputs = vae_net(images)
    loss = criterion(outputs, images, size_average=False)
    kld = 0.5 * torch.mean(vae_net.std.pow(2) + vae_net.mu.pow(2) - torch.log(vae_net.std.pow(2)) - 1.)
    samples = dict(images=dict(generated=outputs, real=images),
                   latents=dict(latent=vae_net.latent.data),
                   labels=dict(latent=inputs['targets'].data))
    return loss + beta_kld * kld, dict(loss=loss.data[0], kld=kld.data[0]), samples, 'reconstruction'


def build_model(data_handler, encoder_args={}, decoder_args={}, vae_args={}):
    shape = data_handler.get_dims('x', 'y', 'c')

    if shape[0] == 64:
        encoder_args_['n_steps'] = 4
        decoder_args_['n_steps'] = 4

    encoder = Encoder(shape, **encoder_args)
    decoder = Decoder(shape, **decoder_args)
    net = Vae(encoder, decoder, nonlinearity=nn.Sigmoid(), **vae_args)
    return dict(vae=net), build_graph