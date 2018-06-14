from . import models

__author__ = 'Bradley Baker'
__author_email__ = 'bbaker@mrn.org'

''' CLIPPING is a global dictionary of clipping boundaries,
    keyed by model name'''
CLIPPING = {}

''' REGULARIZER is a global dictionary of floats, the
    lambda scaling factor for L1 regularization
    keyed by model name'''
L1_DECAY = {}


def init(clipping=None, weight_decay=None):
    '''called in setup.py, initialize clipping and
       weight_decay dicts'''
    global CLIPPING, L1_DECAY
    clipping = clipping or {}
    weight_decay = weight_decay or {}
    CLIPPING.update(**clipping)
    L1_DECAY.update(**weight_decay)


def clip(key):
    '''
       called in train.py, clip weights
    '''
    if key not in CLIPPING:
        return
    bound = CLIPPING[key]
    if key in models.MODEL_HANDLER:
        model = models.MODEL_HANDLER[key]
        if isinstance(model, (list, tuple)):
            for net in model:
                for p in net.parameters():
                    p.data.clamp_(-bound, bound)
        else:
            for p in model.parameters():
                p.data.clamp_(-bound, bound)


def l1_decay(key):
    '''
       called in train.py, do L1 regularization
    '''
    if key not in L1_DECAY:
        return
    factor = L1_DECAY[key]
    if key in models.MODEL_HANDLER:
        model = models.MODEL_HANDLER[key]
        if isinstance(model, (list, tuple)):
            for net in model:
                for p in net.parameters():
                    p.add(factor * (p / p.norm(1)))
        else:
            for p in model.parameters():
                p.add(factor * (p / p.norm(1)))
