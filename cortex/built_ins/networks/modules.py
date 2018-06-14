'''General purpose modules

'''

import torch.nn as nn


class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(*self.shape)


class Pipeline(nn.Module):
    def __init__(self, networks):
        super(Pipeline, self).__init__()
        self.networks = networks

    def forward(self, input):
        output = input
        for network in self.networks:
            output = network(output)
        return output
