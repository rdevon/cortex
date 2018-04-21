'''Module for training.

'''

import logging
import pprint
import sys
import time

import numpy as np
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn

from .data import DATA_HANDLER
from . import exp, viz
from .utils import bad_values, convert_to_numpy, update_dict_of_lists
from .viz import VizHandler, plot


try: input = raw_input #Python2 compatibility
except NameError: pass

logger = logging.getLogger('cortex.util')

OPTIMIZERS = {}
UPDATES = {}
CLIPPING = {}

optimizer_defaults = dict(
    SGD=dict(momentum=0.9, weight_decay=5e-4),
    Adam=dict(betas=(0.5, 0.999))
)


def setup(optimizer=None, learning_rate=None, updates_per_model=None, lr_decay=None, min_lr=None, decay_at_epoch=None,
          clipping=None, weight_decay=None, optimizer_options='default', model_optimizer_options=None):

    global CLIPPING, UPDATES
    model_optimizer_options = model_optimizer_options or {}
    weight_decay = weight_decay or {}
    clipping = clipping or {}

    if optimizer_options == 'default' and optimizer in optimizer_defaults.keys():
        optimizer_options = optimizer_defaults[optimizer]
    updates_per_model = updates_per_model or dict((k, 1) for k in exp.MODELS.keys())
    UPDATES.update(**updates_per_model)
    CLIPPING.update(**clipping)

    if callable(optimizer):
        op = optimizer
    elif hasattr(optim, optimizer):
        op = getattr(optim, optimizer)
    else:
        raise NotImplementedError('Optimizer not supported `{}`'.format(optimizer))

    for k in exp.ROUTINES.keys():
        if k not in exp.MODELS or k == 'extras':
            continue
        model = exp.MODELS[k]
        logger.info('Building optimizer for {}'.format(k))

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

        for p in model_params:
            p.requires_grad = True

        logger.info('Training with {} and optimizer options {}'.format(optimizer, optimizer_options))
        if isinstance(learning_rate, dict):
            eta = learning_rate[k]
        else:
            eta = learning_rate

        if isinstance(weight_decay, dict):
            wd = weight_decay.get(k, 0)
        else:
            wd = weight_decay

        optimizer_options_ = dict((k, v) for k, v in optimizer_options.items())
        if k in model_optimizer_options.keys():
            optimizer_options_.update(**model_optimizer_options)

        optimizer = op(model_params, lr=eta, weight_decay=wd, **optimizer_options_)
        OPTIMIZERS[k] = optimizer

        if k in CLIPPING.keys():
            logger.info('Clipping {} with {}'.format(k, CLIPPING[k]))

    if exp.USE_CUDA:
        cudnn.benchmark = True


def summarize_results(results):
    results_ = {}
    for k, v in results.items():
        if isinstance(v, dict):
            results_[k] = summarize_results(v)
        else:
            if len(v) > 0:
                results_[k] = np.mean(v)
    return results_


def summarize_results_std(results):
    results_ = {}
    for k, v in results.items():
        if isinstance(v, dict):
            results_[k] = summarize_results(v)
        else:
            results_[k] = np.std(v)
    return results_


def train_epoch(epoch, vh, quit_on_bad_values):
    for k, model in exp.MODELS.items():
        if k == 'extras':
            continue
        if isinstance(model, (tuple, list)):
            for net in model:
                net.train()
        else:
            model.train()

    DATA_HANDLER.reset(string='Training (epoch {}): '.format(epoch))
    vh.ignore = True

    results = {'time': dict((rk, []) for rk in exp.MODELS.keys()),
               'losses': dict((rk, []) for rk in exp.MODELS.keys())}

    try:
        while True:
            for rk, routine in exp.ROUTINES.items():
                if rk not in UPDATES:
                    continue
                for _ in range(UPDATES[rk]):
                    DATA_HANDLER.next()

                    for mk, model in exp.MODELS.items():
                        if mk == 'extras':
                            continue
                        if isinstance(model, (list, tuple)):
                            for net in model:
                                for p in net.parameters():
                                    p.requires_grad = (mk == rk)
                        else:
                            for p in model.parameters():
                                p.requires_grad = (mk == rk)
                    if rk != 'extras':
                        OPTIMIZERS[rk].zero_grad()

                    if isinstance(routine, (tuple, list)):
                        routine = routine[0]
                    start_time = time.time()
                    if rk in exp.ARGS['routines'].keys():
                        args = exp.ARGS['routines'][rk]
                    else:
                        args = exp.ARGS['routines']
                    losses = {}
                    results_ = {}
                    routine(DATA_HANDLER, exp.MODELS, losses, results_, vh, **args)
                    bads = bad_values(results_)
                    if bads and quit_on_bad_values:
                        logger.error('Bad values found (quitting): {} \n All:{}'.format(
                            bads, results_))
                        exit(0)

                    if isinstance(losses, dict):
                        if rk in losses:
                            loss = losses[rk]
                        else:
                            loss = None
                    else:
                        loss = losses

                    if loss is not None:
                        results['losses'][rk].append(loss.data[0])
                        loss.backward()
                    end_time = time.time()
                    results['time'][rk].append(end_time - start_time)
                    update_dict_of_lists(results, **results_)

                    if rk != 'extras':
                        OPTIMIZERS[rk].step()

                    if rk in CLIPPING.keys():
                        if rk == 'extras':
                            continue
                        clip = CLIPPING[rk]
                        if rk in exp.MODELS:
                            model = exp.MODELS[rk]
                            if isinstance(model, (list, tuple)):
                                for net in model:
                                    for p in net.parameters():
                                        p.data.clamp_(-clip, clip)
                            else:
                                for p in model.parameters():
                                    p.data.clamp_(-clip, clip)

    except StopIteration:
        pass

    results = summarize_results(results)
    return results


def test_epoch(epoch, vh, return_std=False):
    for k, model in exp.MODELS.items():
        if k == 'extras':
            continue
        if isinstance(model, (tuple, list)):
            for net in model:
                net.eval()
        else:
            model.eval()

    DATA_HANDLER.reset(test=True, string='Evaluating (epoch {}): '.format(epoch))
    results = {'losses': dict((rk, []) for rk in exp.MODELS.keys())}

    routines = exp.ARGS['test_routines']

    vh.ignore = False
    try:
        while True:
            DATA_HANDLER.next()
            for rk, routine in exp.ROUTINES.items():
                if rk not in exp.MODELS:
                    continue
                if isinstance(routine, (tuple, list)):
                    routine = routine[1]
                if rk in routines.keys():
                    args = routines[rk]
                else:
                    args = routines
                results_ = {}
                losses = {}
                routine(DATA_HANDLER, exp.MODELS, losses, results_, vh, **args)
                if rk in losses:
                    results['losses'][rk].append(losses[rk].data[0])
                update_dict_of_lists(results, **results_)
            vh.ignore = True
    except StopIteration:
        pass

    means = summarize_results(results)
    if return_std:
        stds = summarize_results_std(results)

        return means, stds

    return means


def display_results(train_results, test_results, epoch, epochs, epoch_time, total_time):
    print('\n\tEpoch {}/{} took {:.3f}s. Total time: {:.2f}'.format(epoch + 1, epochs, epoch_time, total_time))
    times = train_results.pop('time')
    print('\tAvg update times: ' + ' | '.join(['{}: {:.2f}'.format(k, v) for k, v in times.items()]))
    train_losses = train_results.pop('losses')
    test_losses = test_results.pop('losses')
    print('\tAvg loss: ' + ' | '.join(['{}: {:.2f} / {:.2f}'.format(k, train_losses[k], test_losses[k])
                                       for k in train_losses.keys()]))

    for k in train_results.keys():
        v_train = train_results[k]
        v_test = test_results[k]
        if isinstance(v_train, dict):
            print('\t' + k + ': ' + ' | '.join(['{}: {:.2f} / {:.2f}'.format(k_, v_train[k_], v_test[k_])
                                                for k_ in v_train.keys()]))
        else:
            print('\t{}: {:.2f} / {:.2f}'.format(k, v_train, v_test))


def main_loop(epochs=None, archive_every=None, test_mode=False, quit_on_bad_values=False):
    info = pprint.pformat(exp.ARGS)
    viz.visualizer.text(info, env=exp.NAME, win='info')
    vh = VizHandler()
    total_time = 0.
    if test_mode:
        test_results, test_std = test_epoch('Testing', vh, return_std=True)
        logger.info(' | '.join(
            ['{}: {:.5f}({:.5f})'.format(k, test_results[k], test_std[k]) for k in test_results.keys()]))
        exit(0)

    try:
        for e in range(epochs):
            epoch = exp.INFO['epoch']

            start_time = time.time()

            # TRAINING
            train_results_ = train_epoch(epoch, vh, quit_on_bad_values)
            convert_to_numpy(train_results_)
            update_dict_of_lists(exp.SUMMARY['train'], **train_results_)

            # TESTING
            test_results_ = test_epoch(epoch, vh)
            convert_to_numpy(test_results_)
            update_dict_of_lists(exp.SUMMARY['test'], **test_results_)

            # Finishing up
            epoch_time = time.time() - start_time
            total_time += epoch_time
            display_results(train_results_, test_results_, e, epochs, epoch_time, total_time)
            plot()
            vh.show()
            vh.clear()
            if (archive_every and epoch % archive_every == 0):
                exp.save(prefix=epoch)

            exp.INFO['epoch'] += 1

    except KeyboardInterrupt:
        kill = False
        while True:
            try:
                response = input('Keyboard interrupt. Kill? (Y/N) '
                                     '(or ^c again)')
            except KeyboardInterrupt:
                kill = True
                break
            response = response.lower()
            if response == 'y':
                kill = True
                break
            elif response == 'n':
                print('Cancelling interrupt. Starting epoch over.')
                break
            else:
                print('Unknown response')

        if kill:
            print('Training interrupted')
            exp.save(prefix='interrupted')
            sys.exit(0)

    exp.save(prefix='final')
