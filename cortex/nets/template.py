"""
Template
"""

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

def BUILD(data, models, **kwargs):
    # Create your models here.
    # mymodel = Mymodel(**kwargs)

    # models.update(mymodel=mymodel) # Uncomment this and add your model


# This performs additional setup before train, if necessary.
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