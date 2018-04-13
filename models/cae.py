'''Contractive AutoEncoder model
based partially on
https://github.com/avijit9/Contractive_Autoencoder_in_Pytorch
'''

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .modules.convnets import SimpleConvEncoder as Encoder
from .modules.conv_decoders import SimpleConvDecoder as Decoder
from .modules.regularization import Regularizer
from .modules.modules import View

logger = logging.getLogger('cortex.models' + __name__)

encoder_args_ = dict(dim_h=64, batch_norm=True, n_steps=3, nonlinearity='ReLU')
decoder_args_ = dict(dim_h=64, batch_norm=True, n_steps=3, dim_in=64)
cae_args_ = dict(dim_h=784, n_steps=3, dim_latent=400)

DEFAULTS = dict(
    data=dict(batch_size=dict(train=64, test=640)),
    optimizer=dict(
        optimizer='Adam',
        learning_rate=1e-4,
    ),
    model=dict(encoder_args=encoder_args_,
               decoder_args=decoder_args_,
               cae_args=cae_args_),
    procedures=dict(criterion=F.mse_loss, regularizer='cl'),
    train=dict(
        epochs=200,
        summary_updates=100,
        archive_every=10
    )
)


class CAE(nn.Module):
    def __init__(self, dim_h=64, dim_latent=64, n_steps=3):
        super(CAE, self).__init__()
        self.fc1 = nn.Linear(dim_h, dim_latent, bias=False)  # Encoder
        self.fc2 = nn.Linear(dim_latent, dim_h, bias=False)  # Decoder
        self.relu = nn.ReLU()
        self.dim_latent = dim_latent
        self.dim_h = dim_h
        self.sigmoid = nn.Sigmoid()

    def encoder(self, x):
        h1 = self.relu(self.fc1(x.view(-1, self.dim_h)))
        return h1

    def decoder(self, z):
        h2 = self.sigmoid(self.fc2(z))
        return h2

    def forward(self, x):
            h1 = self.encoder(x)
            self.latent = h1
            h2 = self.decoder(h1)
            return h1, h2


mse_loss = nn.BCELoss(size_average = False)

def loss_function(W, x, recons_x, h, lam):
    """Compute the Contractive AutoEncoder Loss
    Evalutes the CAE loss, which is composed as the summation of a Mean
    Squared Error and the weighted l2-norm of the Jacobian of the hidden
    units with respect to the inputs.
    See reference below for an in-depth discussion:
      #1: http://wiseodd.github.io/techblog/2016/12/05/contractive-autoencoder
    Args:
        `W` (FloatTensor): (N_hidden x N), where N_hidden and N are the
          dimensions of the hidden units and input respectively.
        `x` (Variable): the input to the network, with dims (N_batch x N)
        recons_x (Variable): the reconstruction of the input, with dims
          N_batch x N.
        `h` (Variable): the hidden units of the network, with dims
          batch_size x N_hidden
        `lam` (float): the weight given to the jacobian regulariser term
    Returns:
        Variable: the (scalar) CAE loss
    """
    mse = mse_loss(recons_x, x)
    # Since: W is shape of N_hidden x N. So, we do not need to transpose it as
    # opposed to #1
    dh = h * (1 - h) # Hadamard product produces size N_batch x N_hidden
    # Sum through the input dimension to improve efficiency, as suggested in #1
    w_sum = torch.sum(Variable(W)**2, dim=1)
    # unsqueeze to avoid issues with torch.mv
    w_sum = w_sum.unsqueeze(1) # shape N_hidden x 1
    contractive_loss = torch.sum(torch.mm(dh**2, w_sum), 0)
    return mse + contractive_loss.mul_(lam)


def build_graph(net, inputs, criterion=None, regularizer=None, factor=1.0):
    images = 0.5 * (inputs['images'] + 1.)
    cae_net = net['cae']
    hidden_representation, outputs = cae_net(images)
    W = cae_net.state_dict()['fc1.weight']
    loss = loss_function(W, images.view(-1, 784), outputs,
                         hidden_representation, factor)
    outputs=outputs.view(outputs.size(0), 28, 28)
    samples = dict(images=dict(real=images),
                   latents=dict(latent=cae_net.latent.data),
                   labels=dict(latent=inputs['targets'].data))
    return loss, dict(loss=loss.data[0]), samples, 'reconstruction'


def build_model(data_handler, encoder_args={}, decoder_args={}, cae_args={}):
    net = CAE(**cae_args)
    shape = data_handler.get_dims('x', 'y', 'c')
    return dict(cae=net), build_graph
