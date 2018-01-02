'''Simple AE model

'''

import logging

import torch
from torch import autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from conv_decoders import SimpleConvDecoder as Decoder
from convnets import SimpleConvEncoder as Encoder

logger = logging.getLogger('cortex.models' + __name__)

GLOBALS = {'DIM_X': None, 'DIM_Y': None, 'DIM_C': None, 'DIM_L': None}

encoder_args_ = dict(dim_h=64, batch_norm=True, f_size=5, pad=2, stride=2, min_dim=7,
                           nonlinearity='LeakyReLU', dim_out=10)
decoder_args_ = dict(dim_h=64, batch_norm=True, f_size=5, pad=2, stride=1, dim_in=10)

DEFAULTS = dict(
    data=dict(batch_size=64,
              test_batch_size=64),
    optimizer=dict(
        optimizer='Adam',
        learning_rate=1e-4,
    ),
    model=dict(encoder_args=encoder_args_, decoder_args=decoder_args_),
    procedures=dict(criterion=nn.L1Loss()),
    train=dict(
        epochs=200,
        summary_updates=100,
        archive_every=10
    )
)


def autoencoder(nets, inputs, criterion):
    X = inputs['images']
    encoder = nets['encoder']
    decoder = nets['decoder']
    latent = encoder(X)
    X_prime = decoder(latent, nonlinearity=F.tanh)
    loss = criterion(X_prime, X)

    samples =  dict(images=dict(generated=0.5 * (X_prime.data + 1.), real=0.5 * (X.data + 1.)))
    return loss, dict(loss=loss.data[0]), samples, 'reconstruction'


def build_model(encoder_args={}, decoder_args={}):
    shape = (DIM_X, DIM_Y, DIM_C)

    print(decoder_args)
    encoder = Encoder(shape, **encoder_args)
    decoder = Decoder(shape, **decoder_args)
    logger.debug(encoder)
    logger.debug(decoder)

    return dict(encoder=encoder, decoder=decoder), autoencoder


