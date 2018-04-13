'''
Implements L1,  L2, Elastic-Net, Spectral Regularization, and Contractive Loss
for various loss-types.

TODO L1 with truncation

'''
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable


def contractive_loss(parameters, hidden_u):
    """Compute the Contractive AutoEncoder Loss
        input  `W` (FloatTensor): (N_hidden x N), where N_hidden and N are the
          dimensions of the hidden units and input respectively.

        `h` (Variable): the hidden units of the network, with dims
          batch_size x N_hidden\

         based on
         https://github.com/avijit9/Contractive_Autoencoder_in_Pytorch/
    """
    dh = hidden_u * (1 - hidden_u)  # Hadamard product
    # Sum through the input dimension to improve efficiency, as suggested in #1
    w_sum = torch.sum(Variable(parameters)**2, dim=1)
    # unsqueeze to avoid issues with torch.mv
    w_sum = w_sum.unsqueeze(1)  # shape N_hidden x 1
    print(w_sum)
    print(dh)
    return torch.sum(torch.mm(dh**2, w_sum), 0)


def spectral_norm(input):
    '''
        input : a pytorch tensor, for now, assuming real input
        output : the spectral norm of the tensor, taken by partitions

        TODO: fix for general tensors
    '''
    x = input
    print(x.size())
    if len(x.size()) == 4: # Then it's a convolutional layer
        x = x.transpose(0, 1)
        x = x.contiguous()
        x = x.view(x.size(0), x.size(1)*x.size(2)*x.size(3))
    elif len(x.size()) > 2:
        x = x.view(x.size(0), np.prod(x.size()[1:]))
    elif len(x.size()) == 1:
        x = x.view(1, x.size(0)) 
    ex = torch.eig(x.t()*x)
    ex = ex[:][0]
    print(torch.max(ex))
    return torch.max(ex)


def l1_norm(input):
        '''
            input: a pytorch tensor
            output: l1 norm functional
        '''
        return input.norm(1)


def l2_norm(input):
    '''
        input: a pytorch tensor
        output: l2 norm functional
    '''
    return input.norm(2)


def elastic_net(input):
    '''
        input: a pytorch tensor
        output: elastic net norm functional
    '''
    return l1_norm(input) + l2_norm(input)


class Regularizer():
    '''
        Class for performing regularization on parameters
    '''
    REGULARIZERS = {'l1': l1_norm, 'l2': l2_norm, 'en': elastic_net,
                    'sp': spectral_norm, 'cl': contractive_loss}
    regularizer = None
    factor = 0.0005
    loss = None

    def __init__(self, reg_type='l1', factor=0.0005,
                 loss=nn.CrossEntropyLoss()):
        '''
           reg_type - the kind of regularization
                  l1 - l1 regularization
                  l2 - l2 regularization
                  en - elastic net regularization
                  sp - spectral regularization

          factor - the regularization hyper parameter
          loss - the kind of loss to regularize
          methods: compute_reg_term - see docstring below
        '''
        self.reg_type = reg_type
        self.regularizer = self.REGULARIZERS[reg_type]
        self.factor = factor
        self.loss = loss

    def compute_reg_term(self, input, target, parameters, hidden_units=None):
        '''
            computes the regularization term for given regularization
                input: a torch tensor of inputs
                target: a torch tensor of targets
                parameters: the parameters to regularize
        '''
        total_loss = self.loss(input, target)
        if self.reg_type == 'cl':
            total_loss += self.factor*self.regularizer(parameters,
                                                       hidden_units)
        for beta in parameters:
            total_loss += self.factor*self.regularizer(beta)
        return total_loss
