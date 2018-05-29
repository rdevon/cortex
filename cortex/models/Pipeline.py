import torch.nn as nn

class Pipeline(nn.Module):
    """
    TODO
    """
    def __init__(self, networks):
        super(Pipeline, self).__init__()
        self.networks = networks
    def forward(self, input):
        """
        TODO
        :param input:
        :type input:
        :return:
        :rtype:
        """
        output = input
        for network in self.networks:
            output = network(output)
        return output
