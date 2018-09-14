'''Module for setting up the optimizer.

'''

from collections import defaultdict
import logging

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn

from . import exp


__author__ = 'R Devon Hjelm'
__author_email__ = 'erroneus@gmail.com'

logger = logging.getLogger('cortex.optimizer')
OPTIMIZERS = {}

_optimizer_defaults = dict(
    SGD=dict(),
    Adam=dict(betas=(0.5, 0.999))
)


def wrap_optimizer(C):
    class Op(C):
        def __init__(self, params, clipping=None, **kwargs):
            super().__init__(params, **kwargs)

            if clipping is not None and clipping < 0.0:
                raise ValueError(
                    "Invalid clipping value: {}".format(clipping))

            self.defaults.update(clipping=clipping)

            self.state = defaultdict(dict)
            self.param_groups = []

            param_groups = list(params)
            if len(param_groups) == 0:
                raise ValueError("optimizer got an empty parameter list")
            if not isinstance(param_groups[0], dict):
                param_groups = [{'params': param_groups}]

            for param_group in param_groups:
                self.add_param_group(param_group)

        def step(self, closure=None):
            """Performs a single optimization step.

            Arguments:
                closure (callable, optional): A closure that reevaluates the model
                    and returns the loss.
            """
            loss = super().step(closure=closure)

            for group in self.param_groups:
                bound = group['clipping']
                if bound:
                    for p in group['params']:
                        p.data.clamp_(-bound, bound)
            return loss

    return Op


def setup(model, optimizer='Adam', learning_rate=1.e-4,
          weight_decay={}, clipping={}, optimizer_options={},
          model_optimizer_options={}):
    '''Optimizer entrypoint.

    Args:
        optimizer: Optimizer type. See `torch.optim` for supported optimizers.
        learning_rate: Learning rate.
        updates_per_routine: Updates per routine.
        clipping: If set, this is the clipping for each model.
        weight_decay: If set, this is the weight decay for specified model.
        optimizer_options: Optimizer options.
        model_optimizer_options: Optimizer options for specified model.

    '''

    OPTIMIZERS.clear()
    model_optimizer_options = model_optimizer_options or {}
    weight_decay = weight_decay or {}
    clipping = clipping or {}

    # Set the optimizer options
    if len(optimizer_options) == 0:
        optimizer_options = 'default'
        if not isinstance(optimizer, str):
            optimizer = 'Adam'
    if optimizer_options == 'default'\
            and optimizer in _optimizer_defaults.keys():
        optimizer_options = _optimizer_defaults[optimizer]
    elif optimizer_options == 'default':
        raise ValueError(
            'Default optimizer options for'
            ' `{}` not available.'.format(optimizer))

    # Set the optimizers
    if callable(optimizer):
        op = optimizer
    elif hasattr(optim, optimizer):
        op = getattr(optim, optimizer)
    else:
        raise NotImplementedError(
            'Optimizer not supported `{}`'.format(optimizer))

    for network_key, network in model.nets.items():
        # Set model parameters to cpu or gpu
        network.to(exp.DEVICE)
        # TODO(Devon): is the next line really doing anything?
        if str(exp.DEVICE) == 'cpu':
            pass
        else:
            torch.nn.DataParallel(
                network, device_ids=range(
                    torch.cuda.device_count()))

    model._reset_epoch()
    model.data.reset(make_pbar=False, mode='test')
    model.train_step(_init=True)
    model.visualize(auto_input=True)

    training_nets = model._get_training_nets()

    logger.info('Setting up optimizers for {}'.format(set(training_nets)))

    for network_key in set(training_nets):
        logger.debug('Building optimizer for {}'.format(network_key))
        network = model.nets[network_key]

        if isinstance(network, (tuple, list)):
            params = []
            for net in network:
                params += list(net.parameters())
        else:
            params = list(network.parameters())

        # Needed for reloading.
        for p in params:
            p.requires_grad = True

        # Learning rates
        if isinstance(learning_rate, dict):
            eta = learning_rate[network_key]
        else:
            eta = learning_rate

        # Weight decay
        if isinstance(weight_decay, dict):
            wd = weight_decay.get(network_key, 0)
        else:
            wd = weight_decay

        if isinstance(clipping, dict):
            cl = clipping.get(network_key, None)
        else:
            cl = clipping

        # Update the optimizer options
        optimizer_options_ = dict((k, v) for k, v in optimizer_options.items())
        optimizer_options_.update(weight_decay=wd, clipping=cl, lr=eta)

        if network_key in model_optimizer_options.keys():
            optimizer_options_.update(**model_optimizer_options)

        # Create the optimizer
        op = wrap_optimizer(op)

        optimizer = op(params, **optimizer_options_)
        OPTIMIZERS[network_key] = optimizer

        logger.debug(
            'Training {} routine with {}'.format(
                network_key, optimizer))

    if not exp.DEVICE == torch.device('cpu'):
        cudnn.benchmark = True
