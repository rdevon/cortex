'''
Implements L1,  L2, Elastic-Net and Spectral Regularization
for various loss-types.

TODO L1 with truncation

'''
import torch.nn as nn
import torch.nn.functional as F
import torch

def spectral_norm(input):
   '''
       input : a pytorch tensor, for now, assuming real input

       output : the spectral norm of the tensor
   '''
   return torch.sqrt(torch.max(torch.eig(torch.mul(torch.t(input), input))))
   


def reg_term(input, target, reg="l1", factor=0.0005, **kwargs):
    '''
        reg - computes standard regularization terms
        
        Args:
            input: input tensor (N, *)
            target: target tensor (N, *)

        Output: scalar, if reduce is false

        KWArgs:
            reg: type of regularization: l1, l2, en, or spectral
            factor: \lambda regularization hyper-parameter
            kwargs: <see the nn.L1Loss docstring>
    '''
    reg = reg.lower()
    reg_loss = 0
    if reg == "l1" or reg == "en":
        l1_loss = nn.L1Loss(**kwargs)
        reg_loss += factor*l1_loss(input)
    if reg == "l2" or reg == "en":
        l2_loss = nn.MSELoss(**kwargs)
        reg_loss += factor*l2_loss(input)
    if reg == "spec":
        reg_loss += factor*spectral_norm(input)
    return reg_loss
