"""
General purpose modules
"""

import torch.nn as nn

class View(nn.Module):
    """
    TODO
    """
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(*self.shape)
