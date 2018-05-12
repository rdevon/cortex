'''Template

'''

import torch.nn.functional as F

from classifier import classify
from ali import apply_penalty, build_discriminator as build_mine_discriminator, score
from featnet import get_results
from modules.fully_connected import FullyConnectedNet

from utils import ms_ssim, update_decoder_args, update_encoder_args

def encode(models, X):
    if 'vae' in models:
        models.vae(X)
        Z = models.vae.mu
    else:
        Z = models.encoder(X)

    return Z

# Must have data, models, losses, results, and viz. **kargs should match the keys in DEFAULT_CONFIG.routines below.
def network_routine(data, models, losses, results, viz):
    X, Y = data.get_batch('1.images', '1.targets')
    classifier = models.classifier
    decoder = models.decoder

    Z_P = encode(models, X)
    Z_t = Z_P.data.detach()
    X_d = decoder(Z_t, nonlinearity=F.tanh)
    dd_loss = ((X - X_d) ** 2).sum(1).sum(1).sum(1).mean()
    classify(classifier, Z_P, Y, losses=losses, results=results)

    #correlations = cross_correlation(Z_P, remove_diagonal=True)
    msssim = ms_ssim(X, X_d).item()

    losses.decoder = dd_loss
    results.update(reconstruction_loss=dd_loss.item(), ms_ssim=msssim)
    #viz.add_heatmap(correlations.data, name='latent correlations')
    viz.add_image(X_d, name='Reconstruction')
    viz.add_image(X, name='Ground truth')


def mine_routine(data, models, losses, results, viz, measure='KL', penalty_amount=0.5):
    X_P, X_Q, T = data.get_batch('1.images', '2.images', '1.targets')

    Z = encode(models, X_P)
    E_pos, E_neg, P_samples, Q_samples = score(models, X_P, X_Q, Z, Z, measure, key='mine')
    get_results(P_samples, Q_samples, E_pos, E_neg, measure, results=results, name='mine')

    penalty = apply_penalty(models, losses, results, X_P, Z, penalty_amount, key='mine')

    losses.mine = E_neg - E_pos

    if penalty:
        losses.mine += penalty


# CORTEX ===============================================================================================================
# Must include `BUILD` and `TRAIN_ROUTINES`

def BUILD(data, models, model_type='convnet', mine_args={}, reconstruction_args={}, classifier_args={},
          **kwargs):
    x_shape = data.get_dims('x', 'y', 'c')
    dim_l = data.get_dims('labels')

    # (devon) I don't think this is a great solution.
    data.reset(make_pbar=False)
    data.next()
    X = data.get_batch('1.images')
    if 'vae' in models:
        models.vae(X)
        dim_z = models.vae.mu.size()[1]
    else:
        if 'autoencoder' in models:
            autoencoder = models.pop('autoencoder')
            models.encoder = autoencoder[0]
        elif 'generator' in models:
            generator = models.pop('generator')
            models.encoder = generator[0]
        dim_z = models.encoder(X).size()[1]

    Encoder, mine_args = update_encoder_args(x_shape, model_type=model_type, encoder_args=mine_args)
    Decoder, reconstruction_args = update_decoder_args(x_shape, model_type=model_type, decoder_args=reconstruction_args)

    decoder = Decoder(x_shape, dim_in=dim_z, **reconstruction_args)
    classifier = FullyConnectedNet(dim_z, dim_h=[200, 200], dim_out=dim_l, **classifier_args)
    build_mine_discriminator(models, x_shape, dim_z, Encoder, key='mine', **mine_args)

    models.update(classifier=classifier, decoder=decoder)
    data.reset()


# Dictionary reference to train routines. Keys are up to you
TRAIN_ROUTINES = dict(mine=mine_routine, networks=network_routine)

# Dictionary reference to test routines. If not set, will be copied from train. If value is None, will not be used in test.
TEST_ROUTINES = dict()

# Default configuration for this model
DEFAULT_CONFIG = dict(data=dict(batch_size=dict(train=64, test=640), duplicate=2))