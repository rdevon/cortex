'''
Implements L1,  L2, Elastic-Net and Spectral Regularization
for various loss-types.

TODO L1 with truncation

'''
import torch
import torch.nn as nn


def spectral_norm(input):
    '''
        input : a pytorch tensor, for now, assuming real input
        output : the spectral norm of the tensor, a functional
    '''
    return torch.sqrt(torch.max(torch.eig(torch.mul(torch.t(input), input))))


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
    REGULARIZERS = {'l1': l1_norm, 'l2':l2_norm, 'en':elastic_net, 'sp':spectral_norm}
    regularizer = None
    factor = 0.0005
    loss = None

    def __init__(self, reg_type='l1', factor=0.0005, loss=nn.CrossEntropyLoss()):
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
        self.regularizer = self.REGULARIZERS[reg_type]
        self.factor=factor
        self.loss = loss

    def compute_reg_term(self, input, target, parameters):
        '''
            computes the regularization term for given regularization
                input: a torch tensor of inputs
                target: a torch tensor of targets
                parameters: the parameters to regularize
        '''
        total_loss = self.loss(input, target)
        for beta in parameters:
            total_loss += self.factor*self.regularizer(beta)
        return total_loss

