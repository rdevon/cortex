'''Module for setting up the optimizer.

'''

__author__ = 'R Devon Hjelm'
__author_email__ = 'erroneus@gmail.com'

import logging

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn

from . import exp, reg

logger = logging.getLogger('cortex.optimizer')

UPDATES = {}
OPTIMIZERS = {}
TRAIN_FOR = None

_optimizer_defaults = dict(
    SGD=dict(momentum=0.9, weight_decay=5e-4),
    Adam=dict(betas=(0.5, 0.999))
)

_args = dict(
    optimizer='Adam',
    learning_rate=1e-4,
    updates_per_routine=None,
    train_for=None,
    clipping=None,
    weight_decay=None,
    l1_decay=None,
    optimizer_options='default',
    model_optimizer_options=None
)


def setup(optimizer=None, learning_rate=None, updates_per_routine=None, train_for=None, clipping=None,
          weight_decay=None, l1_decay=None, optimizer_options=None, model_optimizer_options=None):

    global TRAIN_FOR, ROUTINE_MODELS

    model_optimizer_options = model_optimizer_options or {}
    weight_decay = weight_decay or {}
    l1_decay = l1_decay or {}
    clipping = clipping or {}

    # Set the optimizer options
    if optimizer_options == 'default' and optimizer in _optimizer_defaults.keys():
        optimizer_options = _optimizer_defaults[optimizer]
    elif optimizer_options == 'default':
        raise ValueError('Default optimizer options for `{}` not available.'.format(optimizer))

    # Set the number of updates per routine
    updates_per_routine = updates_per_routine or {}
    updates_per_routine = dict((k, (1 if k not in updates_per_routine else updates_per_routine[k]))
                               for k in exp.TRAIN_ROUTINES)
    for k in list(exp.TRAIN_ROUTINES.keys()) + list(exp.FINISH_TRAIN_ROUTINES.keys()):
        if k not in updates_per_routine:
            updates_per_routine[k] = 1

    UPDATES.update(**updates_per_routine)
    TRAIN_FOR = train_for

    # Initialize regularization
    reg.init(clipping=clipping, weight_decay=l1_decay)  # initialize regularization

    # Set the optimizers
    if callable(optimizer):
        op = optimizer
    elif hasattr(optim, optimizer):
        op = getattr(optim, optimizer)
    else:
        raise NotImplementedError('Optimizer not supported `{}`'.format(optimizer))

    for model_key, model in exp.MODEL_HANDLER.items():
        if model_key in ('extras', 'final'):
            continue
        logger.info('Building optimizer for {}'.format(model_key))

        # Set model parameters to cpu or gpu
        if isinstance(model, (tuple, list)):
            model_params = []
            for net in model:
                if exp.USE_CUDA:
                    net.cuda()
                    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
                logger.debug('Getting parameters for {}'.format(net))
                model_params += list(net.parameters())
        else:
            if exp.USE_CUDA:
                model.cuda()
                model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
            model_params = list(model.parameters())

        # Needed for reloading.
        for p in model_params:
            p.requires_grad = True

        # Learning rates
        if isinstance(learning_rate, dict):
            eta = learning_rate[model_key]
        else:
            eta = learning_rate

        # Weight decay
        if isinstance(weight_decay, dict):
            wd = weight_decay.get(model_key, 0)
        else:
            wd = weight_decay

        # Update the optimizer options
        optimizer_options_ = dict((k, v) for k, v in optimizer_options.items())
        if model_key in model_optimizer_options.keys():
            optimizer_options_.update(**model_optimizer_options)

        # Creat the optimizer
        optimizer = op(model_params, lr=eta, weight_decay=wd, **optimizer_options_)
        OPTIMIZERS[model_key] = optimizer

        logger.info('Training {} routine with {}'.format(model_key, optimizer))

        # Additional regularization
        if model_key in reg.CLIPPING.keys():
            logger.info('Clipping {} with {}'.format(model_key, reg.CLIPPING[k]))

        if model_key in reg.L1_DECAY.keys():
            logger.info('L1 Decay {} with {}'.format(model_key, reg.L1_DECAY[k]))

    if exp.USE_CUDA:
        cudnn.benchmark = True