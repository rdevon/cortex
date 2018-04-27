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
from . import exp, viz, reg
from .utils import bad_values, convert_to_numpy, update_dict_of_lists
from .viz import VizHandler, plot


try: input = raw_input #Python2 compatibility
except NameError: pass

logger = logging.getLogger('cortex.util')

OPTIMIZERS = {}
UPDATES = {}
TRAIN_FOR = None

optimizer_defaults = dict(
    SGD=dict(momentum=0.9, weight_decay=5e-4),
    Adam=dict(betas=(0.5, 0.999))
)


def setup(optimizer=None, learning_rate=None, updates_per_model=None, train_for=None,
          clipping=None, weight_decay=None, l1_decay=None, optimizer_options='default', model_optimizer_options=None):

    global TRAIN_FOR

    model_optimizer_options = model_optimizer_options or {}
    weight_decay = weight_decay or {}
    l1_decay = l1_decay or {}
    clipping = clipping or {}

    if optimizer_options == 'default' and optimizer in optimizer_defaults.keys():
        optimizer_options = optimizer_defaults[optimizer]
    updates_per_model = updates_per_model or dict((k, 1) for k in exp.MODELS.keys())
    for k in exp.MODELS.keys():
        if k not in updates_per_model:
            updates_per_model[k] = 1

    UPDATES.update(**updates_per_model)
    TRAIN_FOR = train_for
    reg.init(clipping=clipping, weight_decay=l1_decay)  # initialize regularization

    if callable(optimizer):
        op = optimizer
    elif hasattr(optim, optimizer):
        op = getattr(optim, optimizer)
    else:
        raise NotImplementedError('Optimizer not supported `{}`'.format(optimizer))

    for k in exp.ROUTINES.keys():
        if k not in exp.MODELS or k in ('extras', 'final'):
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

        logger.info('Training {} routine with {}'.format(k, optimizer))

        if k in reg.CLIPPING.keys():
            logger.info('Clipping {} with {}'.format(k, reg.CLIPPING[k]))

        if k in reg.L1_DECAY.keys():
            logger.info('L1 Decay {} with {}'.format(k, reg.L1_DECAY[k]))

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


def train_on_routine(routine_key, quit_on_bad_values, results, viz_handler):
    if routine_key == 'final':
        return
    if routine_key == 'extras':
        OPTIMIZERS[routine_key].zero_grad()

    if routine_key in exp.ARGS['routines'].keys():
        args = exp.ARGS['routines'][routine_key]
    else:
        args = exp.ARGS['routines']

    routine = exp.ROUTINES[routine_key]

    if isinstance(routine, (tuple, list)):
        routine = routine[0]

    # Perform routine
    start_time = time.time()

    losses = {}
    routine_results = {}
    routine(DATA_HANDLER, exp.MODELS, losses, routine_results, viz_handler, **args)
    bads = bad_values(routine_results)

    if bads and quit_on_bad_values:
        logger.error('Bad values found (quitting): {} \n All:{}'.format(
            bads, routine_results))
        exit(0)

    if isinstance(losses, dict):
        if routine_key in losses:
            loss = losses[routine_key]
        else:
            loss = None
    else:
        loss = losses

    # Do backward step
    if loss is not None:
        results['losses'][routine_key].append(loss.item())
        loss.backward()

    end_time = time.time()
    results['time'][routine_key].append(end_time - start_time)
    update_dict_of_lists(results, **routine_results)

    OPTIMIZERS[routine_key].step()

    reg.clip(routine_key)  # weight clipping
    reg.l1_decay(routine_key)  # l1 weight decay


def perform_routine(routine_key, results, viz_handler, test=False):
    if routine_key in exp.ARGS['routines'].keys():
        args = exp.ARGS['test_routines'][routine_key]
    else:
        args = exp.ARGS['test_routines']

    routine = exp.ROUTINES[routine_key]

    if isinstance(routine, (tuple, list)):
        if test:
            routine = routine[1]
        else:
            routine = routine[0]

    routine_results = {}
    losses = {}
    routine(DATA_HANDLER, exp.MODELS, losses, routine_results, viz_handler, **args)
    if routine_key in losses:
        results['losses'][routine_key].append(losses[routine_key].item())
    update_dict_of_lists(results, **routine_results)


def set_updates_dict(epoch):
    if TRAIN_FOR is not None:
        total_steps = sum(TRAIN_FOR.values())
        step = epoch % total_steps
        ts = 0
        for mk, s in TRAIN_FOR.items():
            ts += s
            if step < ts:
                break
        num_updates_dict = dict((k, 0) for k in exp.ROUTINES.keys())
        num_updates_dict[mk] = 1
    else:
        num_updates_dict = UPDATES

    return num_updates_dict


def train_epoch(epoch, viz_handler, quit_on_bad_values):
    for k, model in exp.MODELS.items():
        if k == 'extras':
            continue
        if isinstance(model, (tuple, list)):
            for net in model:
                net.train()
        else:
            model.train()

    DATA_HANDLER.reset(string='Training (epoch {}): '.format(epoch))
    viz_handler.ignore = True

    results = {'time': dict((rk, []) for rk in exp.MODELS.keys()),
               'losses': dict((rk, []) for rk in exp.MODELS.keys())}

    num_updates_dict = set_updates_dict(epoch)

    try:
        while True:
            DATA_HANDLER.next()
            for routine_key in exp.ROUTINES.keys():
                num_updates = num_updates_dict.get(routine_key, 0)
                for u in range(num_updates):
                    if u > 0:
                        DATA_HANDLER.next()
                    train_on_routine(routine_key, quit_on_bad_values, results, viz_handler)
    except StopIteration:
        pass

    if 'final' in exp.ROUTINES:
        assert False
        perform_routine('final', results, viz_handler)

    results = summarize_results(results)

    return results


def test_epoch(epoch, viz_handler, return_std=False):
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

    viz_handler.ignore = False
    try:
        while True:
            DATA_HANDLER.next()
            for routine_key in exp.ROUTINES.keys():
                if routine_key == 'final':
                    continue
                perform_routine(routine_key, results, viz_handler, test=True)
            viz_handler.ignore = True
    except StopIteration:
        pass

    if 'final' in exp.ROUTINES:
        assert False
        perform_routine('final', results, viz_handler, test=True)

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
        v_test = test_results[k] if k in test_results else None
        if v_test is None:
            if isinstance(v_train, dict):
                print('\t' + k + ': ' + ' | '.join(['{}: {:.2f}'.format(k_, v_train[k_])
                                                    for k_ in v_train.keys()]))
            else:
                print('\t{}: {:.2f}'.format(k, v_train))
        else:
            if isinstance(v_train, dict):
                print('\t' + k + ': ' + ' | '.join(['{}: {:.2f} / {:.2f}'.format(k_, v_train[k_], v_test[k_])
                                                    for k_ in v_train.keys()]))
            else:
                print('\t{}: {:.2f} / {:.2f}'.format(k, v_train, v_test))


def align_summaries(d_train, d_test):
    keys = set(d_train.keys()).union(set(d_test.keys()))
    for k in keys:
        if k in d_train and k in d_test:
            v_train = d_train[k]
            v_test = d_test[k]
            if isinstance(v_train, dict):
                max_train_len = max([len(v) for v in v_train.values()])
                max_test_len = max([len(v) for v in v_test.values()])
                max_len = max(max_train_len, max_test_len)
                for k_, v in v_train.items():
                    if len(v) < max_len:
                        v_train[k_] = v_train[k_] + [v_train[k_][-1]] * (max_len - len(v_train[k_]))
                for k_, v in v_test.items():
                    if len(v) < max_len:
                        v_test[k_] = v_test[k_] + [v_test[k_][-1]] * (max_len - len(v_test[k_]))
            else:
                if len(v_train) > len(v_test):
                    d_test[k] = v_test + [v_test[-1]] * (len(v_train) - len(v_test))
                elif len(v_test) > len(v_train):
                    d_train[k] = v_train + [v_train[-1]] * (len(v_test) - len(v_train))
        elif k in d_train:
            v_train = d_train[k]
            if isinstance(v_train, dict):
                max_len = max([len(v) for v in v_train.values()])
                for k_, v in v_train.items():
                    if len(v) < max_len:
                        v_train[k_] = v_train[k_] + [v_train[k_][-1]] * (max_len - len(v_train[k_]))
        elif k in d_test:
            v_test = d_test[k]
            if isinstance(v_test, dict):
                max_len = max([len(v) for v in v_test.values()])
                for k_, v in v_test.items():
                    if len(v) < max_len:
                        v_test[k_] = v_test[k_] + [v_test[k_][-1]] * (max_len - len(v_test[k_]))


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

            align_summaries(exp.SUMMARY['train'], exp.SUMMARY['test'])

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
