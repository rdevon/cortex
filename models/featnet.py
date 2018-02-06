'''Simple GAN model

'''

import logging

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from .gan import f_divergence, apply_penalty
from .modules.densenet import DenseNet


logger = logging.getLogger('cortex.models' + __name__)

resnet_encoder_args_ = dict(dim_h=64, batch_norm=False, f_size=3, n_steps=3)
resnet_decoder_args_ = dict(dim_h=64, batch_norm=True, f_size=3, n_steps=3)
mnist_encoder_args_ = dict(dim_h=64, batch_norm=False, f_size=5, pad=2, stride=2, min_dim=7, nonlinearity='LeakyReLU')
mnist_decoder_args_ = dict(dim_h=64, batch_norm=True, f_size=4, pad=1, stride=2, n_steps=2)
convnet_encoder_args_ = dict(dim_h=64, batch_norm=False, n_steps=3, nonlinearity='ReLU')
convnet_decoder_args_ = dict(dim_h=64, batch_norm=True, n_steps=3)

DEFAULTS = dict(
    data=dict(batch_size=dict(train=64, test=1028), skip_last_batch=True,
              noise_variables=dict(r=('uniform', 16), s=('uniform', 16))),
    optimizer=dict(
        optimizer='Adam',
        learning_rate=dict(discriminator=1.e-4, nets=1e-4, ss_nets=1e-4),
        clipping=dict(discriminator=0.01),
        updates_per_model=dict(discriminator=1, nets=1, ss_nets=1)
    ),
    model=dict(model_type='convnet', dim_e=2, dim_d=16, encoder_args=None),
    procedures=dict(measure='proxy_gan', boundary_seek=True, penalty=1.0),
    train=dict(
        epochs=500,
        summary_updates=100,
        archive_every=10
    )
)


def setup(model=None, data=None, procedure=None, **kwargs):
    #data['noise_variables']['r'] = (data['noise_variables']['r'][0], model['dim_d'])
    data['noise_variables']['r'] = (data['noise_variables']['r'][0], 1)
    data['noise_variables']['s'] = (data['noise_variables']['s'][0], model['dim_d'])


def make_graph(nets, data_handler, measure=None, boundary_seek=False, penalty=None, r_lambda=1., e_lambda=1.,
          noise_scale=1.0):

    # Variables
    X = data_handler['images']
    Y = data_handler['targets']
    Rr = data_handler['r']
    S = data_handler['s']
    S = (S.max(1, keepdim=True)[0] == S).float()
    if Rr.size(1) == 1:
        offsets = 0
    else:
        offsets = Variable(torch.FloatTensor(2 * np.arange(Rr.size(1)) / (Rr.size(1) - 1.) - 1.).cuda())

    Rr = Rr * S

    '''
    Rr =  (Rr + noise_scale * offsets).view(-1, 1) / noise_scale
    Rr = Rr[torch.randperm(Rr.size(0)).cuda()][:X.size(0)]
    '''

    # Nets
    discriminator = nets['discriminator']
    encoder, topnet, revnet =  nets['nets']
    decoder, classifier = nets['ss_nets']

    z = encoder(X)
    z_t = Variable(z.detach().data.cuda())
    X_r = decoder(z_t, nonlinearity=F.tanh)
    y_hat = classifier(z_t, nonlinearity=F.log_softmax)
    Rf = topnet(z, nonlinearity=F.sigmoid)

    z_f = revnet(Rf)
    z_r = revnet(Rr)

    #real_r = discriminator(torch.cat([Rf, z_f], 1))
    real_r = discriminator(torch.cat([Rr, z_r], 1))
    real_f = discriminator(torch.cat([Rf, z], 1))

    d_loss, e_loss, rr, fr, wr, br = f_divergence(measure, real_r, real_f, boundary_seek=boundary_seek)

    #r_loss = (torch.sqrt((z_f - z_t) ** 2)).sum(1).mean()
    r_loss = ((z_f - z_t) ** 2).sum(1).mean()
    c_loss = torch.nn.CrossEntropyLoss()(y_hat, Y)

    predicted = torch.max(y_hat.data, 1)[1]
    correct = 100. * predicted.eq(Y.data).cpu().sum() / Y.size(0)

    dd_loss = ((X - X_r) ** 2).sum(1).sum(1).sum(1).mean()

    results = dict(e_loss=e_loss.data[0], r_loss=r_loss.data[0], d_loss=d_loss.data[0],
                   c_loss=c_loss.data[0], dd_loss=dd_loss.data[0],
                   real=Rr.mean().data[0], real_std=torch.std(Rf).data[0], accuracy=correct)
    samples = dict(scatters=dict(real=(z.data, Y.data)),
                   histograms=dict(encoder=dict(fake=Rf.max(1)[1].data, real=Rr.max(1)[1].data)),
                   images=dict(reconstruction=0.5 * (X_r.data + 1.), original=0.5 * (X.data + 1.)))

    if penalty:
        p_term = apply_penalty(data_handler, discriminator, Rr, Rf, measure)

        d_loss += penalty * p_term.mean()
        results['real gradient penalty'] = p_term.mean()

    loss = dict(nets=(e_lambda*e_loss)+(r_lambda*r_loss), discriminator=d_loss, ss_nets=(dd_loss+c_loss))
    return loss, results, samples, 'boundary'


def build_model(data_handler, model_type='convnet', dim_d=16, dim_e=2, encoder_args=None, decoder_args=None):
    encoder_args = encoder_args or {}
    decoder_args = decoder_args or {}
    shape = data_handler.get_dims('x', 'y', 'c')
    dim_l = data_handler.get_dims('labels')[0]

    if model_type == 'resnet':
        from .modules.resnets import ResEncoder as Encoder
        from .modules.resnets import ResDecoder as Decoder
        encoder_args_ = resnet_encoder_args_
        decoder_args_ = resnet_decoder_args_
    elif model_type == 'convnet':
        from .modules.convnets import SimpleConvEncoder as Encoder
        from .modules.conv_decoders import SimpleConvDecoder as Decoder
        encoder_args_ = convnet_encoder_args_
        decoder_args_ = convnet_decoder_args_
    elif model_type == 'mnist':
        from .modules.convnets import SimpleConvEncoder as Encoder
        from .modules.conv_decoders import SimpleConvDecoder as Decoder
        encoder_args_ = mnist_encoder_args_
        decoder_args_ = mnist_decoder_args_
    else:
        raise NotImplementedError(model_type)

    encoder_args_.update(**encoder_args)
    decoder_args_.update(**decoder_args)

    if shape[0] == 64:
        encoder_args_['n_steps'] = 4
        decoder_args_['n_steps'] = 4

    encoder = Encoder(shape, dim_out=dim_e, fully_connected_layers=[256], **encoder_args_)
    decoder = Decoder(shape, dim_in=dim_e, **decoder_args_)
    classifier = DenseNet(dim_e, dim_h=[64, 64], dim_out=dim_l, batch_norm=True)
    topnet = DenseNet(dim_e, dim_h=[64, 64], dim_out=dim_d, batch_norm=True)
    revnet = DenseNet(dim_d, dim_h=[64, 64], dim_out=dim_e, batch_norm=True)
    discriminator = DenseNet(dim_d+dim_e, dim_h=[64, 64], dim_out=1, nonlinearity='ReLU', batch_norm=False)
    logger.debug(discriminator)
    logger.debug(encoder)

    return dict(discriminator=discriminator, nets=[encoder, topnet, revnet], ss_nets=[decoder, classifier]), make_graph


