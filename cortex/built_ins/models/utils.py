'''Model misc utilities.

'''

import logging
import math

from sklearn import svm
import torch


logger = logging.getLogger('cortex.arch' + __name__)


def log_sum_exp(x, axis=None):
    x_max = torch.max(x, axis)[0]
    y = torch.log((torch.exp(x - x_max)).sum(axis)) + x_max
    return y


def cross_correlation(X, remove_diagonal=False):
    X_s = X / X.std(0)
    X_m = X_s - X_s.mean(0)
    b, dim = X_m.size()
    correlations = (X_m.unsqueeze(2).expand(b, dim, dim) *
                    X_m.unsqueeze(1).expand(b, dim, dim)).sum(0) / float(b)
    if remove_diagonal:
        Id = torch.eye(dim)
        Id = torch.autograd.Variable(Id.cuda(), requires_grad=False)
        correlations -= Id

    return correlations


def perform_svc(X, Y, clf=None):
    if clf is None:
        clf = svm.LinearSVC()
        clf.fit(X, Y)

    Y_hat = clf.predict(X)

    return clf, Y_hat


def ms_ssim(X_a, X_b, window_size=11, size_average=True, C1=0.01**2, C2=0.03**2):
    '''
    Taken from Po-Hsun-Su/pytorch-ssim
    '''

    channel = X_a.size(1)

    def gaussian(sigma=1.5):
        gauss = torch.Tensor(
            [math.exp(-(x - window_size // 2) **
                      2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    def create_window():
        _1D_window = gaussian(window_size).unsqueeze(1)
        _2D_window = _1D_window.mm(
            _1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = torch.Tensor(
            _2D_window.expand(channel, 1, window_size,
                              window_size).contiguous())
        return window.cuda()

    window = create_window()

    mu1 = torch.nn.functional.conv2d(X_a, window,
                                     padding=window_size // 2, groups=channel)
    mu2 = torch.nn.functional.conv2d(X_b, window,
                                     padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = torch.nn.functional.conv2d(
        X_a * X_a, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = torch.nn.functional.conv2d(
        X_b * X_b, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = torch.nn.functional.conv2d(
        X_a * X_b, window, padding=window_size // 2, groups=channel) - mu1_mu2

    ssim_map = (((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) /
                ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


resnet_encoder_args_ = dict(dim_h=64, batch_norm=True, f_size=3, n_steps=3)
mnist_encoder_args_ = dict(dim_h=64, batch_norm=True, f_size=5,
                           pad=2, stride=2, min_dim=7)
convnet_encoder_args_ = dict(dim_h=64, batch_norm=True, n_steps=3)


def update_encoder_args(x_shape, model_type='convnet', encoder_args=None):
    encoder_args = encoder_args or {}
    if model_type == 'resnet':
        from cortex.built_ins.networks.resnets import ResEncoder as Encoder
        encoder_args_ = {k: v for k, v in resnet_encoder_args_.items()}
    elif model_type == 'convnet':
        from cortex.built_ins.networks.convnets import SimpleConvEncoder as Encoder
        encoder_args_ = {k: v for k, v in convnet_encoder_args_.items()}
    elif model_type == 'mnist':
        from cortex.built_ins.networks.convnets import SimpleConvEncoder as Encoder
        encoder_args_ = {k: v for k, v in mnist_encoder_args_.items()}
    elif model_type.split('.')[0] == 'tv':
        from cortex.built_ins.networks.torchvision import models
        model_attributes = model_type.split('.')
        if len(model_attributes) != 2:
            raise ValueError('`tvr` model type should be in form `tv.<MODEL>`')
        model_key = model_attributes[1]

        try:
            tv_model = getattr(models, model_key)
        except AttributeError:
            raise NotImplementedError(model_attributes[1])

        # TODO This lambda function is necessary because Encoder takes shape
        # and dim_out.
        Encoder = (lambda shape, dim_out=None, n_steps=None,
                   **kwargs: tv_model(num_classes=dim_out, **kwargs))
        encoder_args_ = {}
    elif model_type.split('.')[0] == 'tv-wrapper':
        from cortex.built_ins.networks import tv_models_wrapper as models
        model_attributes = model_type.split('.')

        if len(model_attributes) != 2:
            raise ValueError(
                '`tv-wrapper` model type should be in form'
                ' `tv-wrapper.<MODEL>`')
        model_key = model_attributes[1]

        try:
            Encoder = getattr(models, model_key)
        except AttributeError:
            raise NotImplementedError(model_attributes[1])
        encoder_args_ = {}
    else:
        raise NotImplementedError(model_type)

    encoder_args_.update(**encoder_args)
    if x_shape[0] == 64:
        encoder_args_['n_steps'] = 4
    elif x_shape[0] == 128:
        encoder_args_['n_steps'] = 5

    return Encoder, encoder_args_


resnet_decoder_args_ = dict(dim_h=64, batch_norm=True, f_size=3, n_steps=3)
mnist_decoder_args_ = dict(dim_h=64, batch_norm=True, f_size=4,
                           pad=1, stride=2, n_steps=2)
convnet_decoder_args_ = dict(dim_h=64, batch_norm=True, n_steps=3)


def update_decoder_args(x_shape, model_type='convnet', decoder_args=None):
    decoder_args = decoder_args or {}

    if model_type == 'resnet':
        from cortex.built_ins.networks.resnets import ResDecoder as Decoder
        decoder_args_ = {k: v for k, v in resnet_decoder_args_.items()}
    elif model_type == 'convnet':
        from cortex.built_ins.networks.conv_decoders import (
            SimpleConvDecoder as Decoder)
        decoder_args_ = {k: v for k, v in convnet_decoder_args_.items()}
    elif model_type == 'mnist':
        from cortex.built_ins.networks.conv_decoders import (
            SimpleConvDecoder as Decoder)
        decoder_args_ = {k: v for k, v in mnist_decoder_args_.items()}
    else:
        raise NotImplementedError(model_type)

    decoder_args_.update(**decoder_args)
    if x_shape[0] >= 64:
        decoder_args_['n_steps'] = 4
    elif x_shape[0] == 128:
        decoder_args_['n_steps'] = 5

    return Decoder, decoder_args_


def to_one_hot(y, K):
    y_ = torch.unsqueeze(y, 1).long()

    one_hot = torch.zeros(y.size(0), K).cuda()
    one_hot.scatter_(1, y_.data.cuda(), 1)
    return torch.tensor(one_hot)
