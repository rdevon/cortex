'''Simple GAN model

'''

import logging
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import autograd
from torch.autograd import Variable

from .gan import f_divergence, apply_penalty
from .modules.densenet import DenseNet


logger = logging.getLogger('cortex.models' + __name__)

resnet_encoder_args_ = dict(dim_h=64, batch_norm=True, f_size=3, n_steps=3)
resnet_decoder_args_ = dict(dim_h=64, batch_norm=True, f_size=3, n_steps=3)
mnist_encoder_args_ = dict(dim_h=64, batch_norm=True, f_size=5, pad=2, stride=2, min_dim=7)
mnist_decoder_args_ = dict(dim_h=64, batch_norm=True, f_size=4, pad=1, stride=2, n_steps=2)
convnet_encoder_args_ = dict(dim_h=64, batch_norm=True, n_steps=3)
convnet_decoder_args_ = dict(dim_h=64, batch_norm=True, n_steps=3)

DEFAULTS = dict(
    data=dict(batch_size=dict(train=64, test=1028), skip_last_batch=True,
              noise_variables=dict(r=('uniform', 16), s=('uniform', 16))),
    optimizer=dict(
        optimizer='Adam',
        learning_rate=dict(discriminator=1e-4, nets=1e-4, ss_nets=1e-4),
        #clipping=dict(discriminator=0.1),
        updates_per_model=dict(discriminator=5, nets=1, ss_nets=1)
    ),
    model=dict(model_type='convnet', dim_e=2, dim_d=16, encoder_args=None),
    procedures=dict(measure='gan', boundary_seek=False, penalty=0.),
    train=dict(
        epochs=500,
        summary_updates=100,
        archive_every=10
    )
)


def setup(model=None, data=None, procedures=None, **kwargs):
    #data['noise_variables']['r'] = (data['noise_variables']['r'][0], model['dim_d'])

    data['noise_variables']['s'] = (data['noise_variables']['s'][0], model['dim_d'])

    if procedures['noise_type'] == 'gaussians':
        data['noise_variables']['r'] = ('normal', 1)
        data['noise_variables']['v'] = ('normal', model['dim_d'])
    else:
        data['noise_variables']['r'] = (data['noise_variables']['r'][0], 1)
        data['noise_variables']['v'] = (data['noise_variables']['s'][0], model['dim_d'])

    data['noise_variables']['u'] = (data['noise_variables']['r'][0], 1)
    data['noise_variables']['n_e'] = ('normal', model['dim_e'])
    data['noise_variables']['n_d'] = ('normal', model['dim_e'])

    if procedures['noise_type'] == 'discrete':
        model['dim_d'] += 1


def to_one_hot(y, K):
    y_ = torch.unsqueeze(y, 2)

    one_hot = torch.FloatTensor(y.size(0), y.size(1), K).zero_().cuda()
    one_hot.scatter_(2, y_.data.cuda(), 1)
    return Variable(one_hot)


def make_graph(nets, data_handler, measure=None, boundary_seek=False, penalty=None, r_lambda=1., e_lambda=1.,
               noise_type='lines', n_samples=10, dim_s=128):

    # Variables
    X, Y, Rr, Rs, S, V, Ne, Nd = data_handler.get_batch('images', 'targets', 'r', 'u', 's', 'v', 'n_e', 'n_d')
    batch_size = X.size(0)

    b = np.zeros(S.size(1))

    if noise_type == 'lines':
        offset = 0.
        scale = 1.
        Rr *= scale
        Rr += offset
        b[0] = 1
        a = (scale * (np.arange(64) / 64.) + offset)[:, None] * b[None, :]
        S = (S.max(1, keepdim=True)[0] == S).float()
        Rr = Rr * S
    elif noise_type == 'planes':
        b[0] = 1
        b[1] = 1
        offset = 0.
        scale = 1.
        Rr *= scale
        Rr += offset
        a = (scale * (np.arange(64) / 64.) + offset)[:, None] * b[None, :]
        if dim_s == S.size(1):
            Rr = S
        else:
            assert False
            S_ = to_one_hot(Variable(torch.multinomial(torch.zeros(Rr.size(0), S.size(1)) + 1. / float(S.size(1)), dim_s).cuda()), S.size(1)).sum(1)
            Rr = S * S_
    elif noise_type == 'sparse':
        b[0] = 1
        a = (np.arange(64) / 64.)[:, None] * b[None, :]
        S = (S < 0.1).float()
        Rr = V * S
    elif noise_type == 'gaussians':
        b[0] = 1
        #a = ((np.arange(64) / 64.) - 0.5)[:, None] * b[None, :]
        a = ((np.arange(64) / 64.))[:, None] * b[None, :]
        #S = (S.max(1, keepdim=True)[0] == S).float()
        #Rr = Rr * S
        S = (S < 0.1).float()
        Rr = V * S
    elif noise_type == 'discrete':
        S = (S.max(1, keepdim=True)[0] == S).float()
        Rr = torch.cat([Rr, S], 1)
    elif noise_type == 'unit_sphere':
        Rr = S / S.sum(1, keepdim=True)
        a = (np.arange(64) / 64.)[:, None] * b[None, :]

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
    Rf = topnet(z + Ne)

    def log_sum_exp(x, axis=None, keepdim=False):
        x_max = torch.max(x, axis)[0]
        y = torch.log((torch.exp(x - x_max)).sum(axis, keepdim=keepdim)) + x_max
        return y

    if noise_type == 'discrete':
        logits = Rf[:, 1:]
        values = Rf[:, 0]
        K = logits.size(1)
        probs = F.softmax(logits, dim=1)
        samples = to_one_hot(torch.multinomial(probs, n_samples), K).float().t()

        values_e = values.unsqueeze(0).expand(n_samples, -1)
        Rf = torch.cat([values_e.unsqueeze(2), samples], 2)
        Rf_ = Variable(Rf.data.cuda(), volatile=True)

        try:
            z_f = revnet(Rf[0])
        except:
            print(X.size())
            print(z.size())
            print(logits.size())
            print(values.size())
            print(K)
            print(probs.size())
            print(samples.size())
            print(values_e.size())
            raise
        z_e = z.unsqueeze(0).expand(n_samples, -1, -1)
        z_e_ = Variable(z_e.data.cuda(), volatile=True)
        real_f = discriminator(torch.cat([Rf, z_e], 2).view(n_samples * batch_size, -1))
        real_f_ = discriminator(torch.cat([Rf_, z_e_], 2).view(n_samples * batch_size, -1))

    else:
        z_f = revnet(Rf)

        #w_f = discriminator(Rf)
        #real_f = top_disc(torch.cat([w_f, z], 1))
        real_f = discriminator(torch.cat([Rf, z], 1))

    #Rr = Rr.view(*Rf.size())
    z_r = revnet(Rr)
    real_r = discriminator(torch.cat([Rr, z_r], 1))
    #w_r = discriminator(Rr)
    #real_r = top_disc(torch.cat([w_r, z_r], 1))

    d_loss, e_loss, rr, fr, wr, br = f_divergence(measure, real_r, real_f, boundary_seek=boundary_seek)

    if noise_type == 'discrete':
        lse = log_sum_exp(logits.t(), axis=0, keepdim=True).t().expand(-1, K)
        log_g = (samples * (logits - lse).unsqueeze(0).expand(n_samples, -1, -1)).sum(2)

        log_M = math.log(n_samples)
        log_w = Variable(real_f_.data.cuda(), requires_grad=False).view(n_samples, batch_size)
        log_alpha = log_sum_exp(log_w - log_M, axis=0)

        log_Z_est = log_alpha
        log_w_tilde = log_w - log_Z_est - log_M
        w_tilde = torch.exp(log_w_tilde)
        e_loss -= (w_tilde * log_g).sum(0).mean()
        Rf = Rf[0]

    r_loss = ((z_f - z_t) ** 2).sum(1).mean()
    c_loss = torch.nn.CrossEntropyLoss()(y_hat, Y)

    predicted = torch.max(y_hat.data, 1)[1]
    correct = 100. * predicted.eq(Y.data).cpu().sum() / Y.size(0)

    dd_loss = ((X - X_r) ** 2).sum(1).sum(1).sum(1).mean()

    y = torch.autograd.Variable(torch.FloatTensor(a).cuda())
    x_y = decoder(revnet(y), nonlinearity=F.tanh)
    x_s = decoder(revnet(Rr), nonlinearity=F.tanh)

    results = dict(e_loss=e_loss.data[0], r_loss=r_loss.data[0], d_loss=d_loss.data[0],
                   c_loss=c_loss.data[0], dd_loss=dd_loss.data[0],
                   real=Rr.mean().data[0], real_std=torch.std(Rf).data[0], accuracy=correct)
    samples = dict(scatters=dict(labels=(z.data, Y.data), cluster=(z.data, Rf.max(1)[1] + 1)),
                   histograms=dict(encoder=dict(fake=Rf.max(1)[1].data, real=Rr.max(1)[1].data)),
                   images=dict(reconstruction=0.5 * (X_r.data + 1.), original=0.5 * (X.data + 1.),
                               samples=0.5 * (x_s.data + 1.), interpolation=0.5 * (x_y.data + 1.)))

    if penalty:
        X_ = Variable(X.data.cuda(), requires_grad=True)
        Rr_ = Variable(Rr.data.cuda(), requires_grad=True)
        z_ = encoder(X_)
        Rf_ = topnet(z_)
        z_r_ = revnet(Rr_)

        real = torch.cat([Rr_, z_r_], 1)
        fake = torch.cat([Rf_, z_], 1)
        real_out = discriminator(real)
        fake_out = discriminator(fake)
        '''

        w_r_ = discriminator(Rr_)
        w_f_ = discriminator(Rf_)
        real_out = top_disc(torch.cat([w_r_, z_r_], 1))
        fake_out = top_disc(torch.cat([w_f_, z_], 1))
        '''

        g_r = autograd.grad(outputs=real_out, inputs=real, grad_outputs=torch.ones(real_out.size()).cuda(),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]

        g_f = autograd.grad(outputs=fake_out, inputs=fake, grad_outputs=torch.ones(fake_out.size()).cuda(),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]

        g_r = g_r.view(g_r.size()[0], -1)
        g_f = g_f.view(g_f.size()[0], -1)

        g_r = (g_r ** 2).sum(1)
        g_f = (g_f ** 2).sum(1)

        p_term = 0.5 * (g_r.mean() + g_f.mean())

        d_loss += penalty * p_term.mean()
        e_loss += penalty * p_term.mean()
        results['real gradient penalty'] = p_term.mean().data[0]

    loss = dict(nets=(e_lambda*e_loss)+(r_lambda*r_loss), discriminator=d_loss, ss_nets=(dd_loss+c_loss))
    return loss, results, samples, 'boundary'


def build_model(data_handler, model_type='convnet', dim_d=16, dim_e=2, dim_h=512, n_steps=3, encoder_args=None, decoder_args=None):
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

    encoder = Encoder(shape, dim_out=dim_e, fully_connected_layers=[1028], **encoder_args_)
    decoder = Decoder(shape, dim_in=dim_e, **decoder_args_)
    dim_h = [1028, 512, 256, 128]
    classifier = DenseNet(dim_e, dim_h=[64, 64], dim_out=dim_l, batch_norm=True, dropout=0.2)
    topnet = DenseNet(dim_e, dim_h=dim_h[::-1], dim_out=dim_d, batch_norm=True)
    #topnet = Decoder(shape, dim_in=dim_e, **decoder_args_)
    #revnet = Encoder(shape, dim_out=dim_e, **encoder_args_)
    revnet = DenseNet(dim_d, dim_h=dim_h, dim_out=dim_e, batch_norm=True)
    discriminator = DenseNet(dim_d+dim_e, dim_h=dim_h, dim_out=1, batch_norm=False)
    #discriminator = Encoder(shape, dim_out=dim_e, **encoder_args_)
    #top_disc = DenseNet(2*dim_e, dim_out=1, batch_norm=False)
    logger.debug(discriminator)
    logger.debug(encoder)

    return dict(discriminator=discriminator, nets=[encoder, topnet, revnet], ss_nets=[decoder, classifier]), make_graph


