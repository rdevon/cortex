'''Module for Sobel transformation

'''

import torch
import torch.nn.functional as F


class Sobel(object):
    def __init__(self):
        self.kernel_g_x = torch.FloatTensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).unsqueeze(0).unsqueeze(0)
        self.kernel_g_y = torch.FloatTensor(
            [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).unsqueeze(0).unsqueeze(0)

    def _apply_sobel(self, channel):
        g_x = F.conv2d(channel, self.kernel_g_x, stride=1, padding=1)
        g_y = F.conv2d(channel, self.kernel_g_y, stride=1, padding=1)
        return torch.sqrt(torch.pow(g_x, 2) + torch.pow(g_y, 2))

    def __call__(self, img):
        a = torch.cat([self._apply_sobel(
            channel.unsqueeze(0).unsqueeze(0)) for channel in img]).squeeze(1)
        return a

    def __repr__(self):
        return self.__class__.__name__ + '()'
