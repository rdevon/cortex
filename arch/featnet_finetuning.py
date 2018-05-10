'''Template

'''

from .ali import build_discriminator as build_mine_discriminator
from .modules.fully_connected import FullyConnectedNet
from .vae import update_decoder_args, update_encoder_args

# Must have data, models, losses, results, and viz. **kargs should match the keys in DEFAULT_CONFIG.routines below.
def routine(data, models, losses, results, viz, **kwargs):
    # Get data
    X = data['images']

    # Use the key from BUILD.
    net = models.mymodel

    # Your code here:
    output = net(X)
    loss = (output ** 2).mean()
    # End your code

    # Add losses
    losses.mymodel = loss
    # Plots
    results.update(aplot=output.mean().item())
    # Images
    viz.add_image(X, name='An image')


# CORTEX ===============================================================================================================
# Must include `BUILD` and `TRAIN_ROUTINES`

def BUILD(data, models, dim_z=None, dim_embedding=None, dim_noise=None, model_type=None, decoder_args=None,
          encoder_args=None, **kwargs):
    x_shape = data.get_dims('x', 'y', 'c')
    dim_l = data.get_dims('labels')

    dim_embedding = dim_z or dim_embedding or dim_noise

    Encoder, encoder_args = update_encoder_args(x_shape, model_type=model_type, encoder_args=encoder_args)
    Decoder, decoder_args = update_decoder_args(x_shape, model_type=model_type, decoder_args=decoder_args)

    decoder = Decoder(x_shape, dim_in=dim_embedding, **decoder_args)
    classifier = FullyConnectedNet(dim_z, dim_h=[200, 200], dim_out=dim_l, dropout=0.2, batch_norm=True)
    build_mine_discriminator(models, x_shape, dim_embedding, Encoder, key='mine', **encoder_args)

    models.update(classifier=classifier, decoder=decoder)

    assert False, models


# This performs additional setup before training, if necessary.
def SETUP(**kwargs):
    pass


# Dictionary reference to train routines. Keys are up to you
TRAIN_ROUTINES = dict(main_routine=routine)

# Dictionary reference to test routines. If not set, will be copied from train. If value is None, will not be used in test.
TEST_ROUTINES = dict()

# Default configuration for this model
DEFAULT_CONFIG = dict(
    data=dict(),
    optimizer=dict(),
    model=dict(),
    routines=dict(), # this can be a dict or a dict of dicts, where the keys are the routine keys above.
    train=dict()
)