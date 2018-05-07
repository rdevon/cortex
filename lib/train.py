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

from . import data, exp, viz, reg
from .utils import bad_values, convert_to_numpy, Handler, update_dict_of_lists
from .viz import VizHandler, plot


try:
    input = raw_input #Python2 compatibility
except NameError:
    pass

logger = logging.getLogger('cortex.util')

OPTIMIZERS = {}
UPDATES = {}
TRAIN_FOR = None
ROUTINE_MODELS = {}

optimizer_defaults = dict(
    SGD=dict(momentum=0.9, weight_decay=5e-4),
    Adam=dict(betas=(0.5, 0.999))
)


class LossHandler(Handler):
    '''
    Simple dict-like container for losses
    '''

    _type = torch.Tensor
    _get_error_string = 'Loss `{}` not found. You must add it as a dict entry'

    def check_key_value(self, k, v):
        super().check_key_value(k, v)
        if k not in exp.MODEL_HANDLER:
            raise AttributeError('Keyword `{}` not in the model_handler. Found: {}.'.format(
                k, tuple(exp.MODEL_HANDLER.keys())))

    def __setitem__(self, k, v):
        self.check_key_value(k, v)
        if len(v.size()) > 0:
            raise ValueError('Loss size must be a scalar. Got {}'.format(v.size()))
        super().__setitem__(k, v)


def setup(optimizer=None, learning_rate=None, updates_per_routine=None, train_for=None,
          clipping=None, weight_decay=None, l1_decay=None, optimizer_options='default', model_optimizer_options=None):

    global TRAIN_FOR, ROUTINE_MODELS

    model_optimizer_options = model_optimizer_options or {}
    weight_decay = weight_decay or {}
    l1_decay = l1_decay or {}
    clipping = clipping or {}

    # Set the optimizer options
    if optimizer_options == 'default' and optimizer in optimizer_defaults.keys():
        optimizer_options = optimizer_defaults[optimizer]
    elif optimizer_options == 'default':
        raise ValueError('Default optimizer options for `{}` not available.'.format(optimizer))

    # Set the number of updates per routine
    updates_per_routine = updates_per_routine or dict((k, 1) for k in exp.TRAIN_ROUTINES.keys())
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

    # Test the routines and recover the loss keys
    with torch.no_grad():
        data.DATA_HANDLER.reset(make_pbar=False)
        data.DATA_HANDLER.next()
        routine_models = {}
        args = exp.ARGS['routines']

        for routine_key, routine in exp.TRAIN_ROUTINES.items():
            logger.info('Testing routine `{}`'.format(routine_key))
            loss_handler = LossHandler()
            perform_routine(routine_key, routine, loss_handler, {}, VizHandler(), args)
            routine_models[routine_key] = tuple(loss_handler.keys())
    ROUTINE_MODELS.update(**routine_models)


def summarize_results(results):
    results_ = {}
    for k, v in results.items():
        if isinstance(v, dict):
            results_[k] = summarize_results(v)
        else:
            if len(v) > 0:
                try:
                    results_[k] = np.mean(v)
                except:
                    raise ValueError('Something is wrong with result {} of type {}.'.format(k, type(v[0])))

    return results_


def summarize_results_std(results):
    results_ = {}
    for k, v in results.items():
        if isinstance(v, dict):
            results_[k] = summarize_results(v)
        else:
            results_[k] = np.std(v)
    return results_


def train_on_routine(routine_key, routine, loss_handler, results, viz_handler, args, quit_on_bad_values=False):
    for model_key in ROUTINE_MODELS[routine_key]:
        OPTIMIZERS[model_key].zero_grad()

    # Set requires grad from training models.
    for mk, model in exp.MODEL_HANDLER.items():
        if isinstance(model, (list, tuple)):
            for net in model:
                for p in net.parameters():
                    p.requires_grad = mk in ROUTINE_MODELS[routine_key]
        else:
            for p in model.parameters():
                p.requires_grad = mk in ROUTINE_MODELS[routine_key]

    # Perform routine
    start_time = time.time()
    perform_routine(routine_key, routine, loss_handler, results, viz_handler, args,
                    quit_on_bad_values=quit_on_bad_values)

    for model_key, loss in loss_handler.items():
        # Do backward step
        if loss is not None:
            loss.backward()
            OPTIMIZERS[model_key].step()

    end_time = time.time()
    results['time'][routine_key].append(end_time - start_time)


def perform_routine(routine_key, routine, loss_handler, results, viz_handler, args, quit_on_bad_values=False):
    if routine is None:
        return

    if isinstance(args, dict) and routine_key in args:
        args = args[routine_key]

    # Run routine
    routine_results = {}
    routine(data.DATA_HANDLER, exp.MODEL_HANDLER, loss_handler, routine_results, viz_handler, **args)

    # Check for bad numbers
    bads = bad_values(routine_results)
    if bads and quit_on_bad_values:
        logger.error('Bad values found (quitting): {} \n All:{}'.format(bads, routine_results))
        exit(0)

    # Update results
    update_dict_of_lists(results, **routine_results)


def set_updates_dict(epoch):
    routines = {}
    routines.update(**exp.TRAIN_ROUTINES)
    routines.update(**exp.FINISH_TRAIN_ROUTINES)
    if TRAIN_FOR is not None:
        total_steps = sum(TRAIN_FOR.values())
        step = epoch % total_steps
        ts = 0
        for mk, s in TRAIN_FOR.items():
            ts += s
            if step < ts:
                break
        num_updates_dict = dict((k, 0) for k in routines.keys())
        num_updates_dict[mk] = 1
    else:
        num_updates_dict = UPDATES

    return num_updates_dict


def train_epoch(epoch, viz_handler, quit_on_bad_values):
    for k, model in exp.MODEL_HANDLER.items():
        if isinstance(model, (tuple, list)):
            for net in model:
                net.train()
        else:
            model.train()

    results = {'time': dict((rk, []) for rk in exp.TRAIN_ROUTINES.keys()),
               'losses': dict((mk, []) for mk in exp.MODEL_HANDLER.keys())}

    num_updates_dict = set_updates_dict(epoch)
    is_training = ', '.join([k for k in num_updates_dict.keys() if num_updates_dict[k] > 0])
    data.DATA_HANDLER.reset(string='Training (epoch {}) ({}): '.format(epoch, is_training))
    viz_handler.ignore = True
    routine_args = exp.ARGS['routines']

    try:
        while True:
            # Iterate data
            data.DATA_HANDLER.next()

            # Loop through routines
            losses = {}
            for routine_key, routine in exp.TRAIN_ROUTINES.items():
                num_updates = num_updates_dict.get(routine_key, 0)
                routine_losses = {}

                loss_handler = LossHandler()
                for u in range(num_updates):
                    if u > 0:
                        # Iterate more data if this is not the first update
                        data.DATA_HANDLER.next()
                        loss_handler = LossHandler()

                    train_on_routine(routine_key, routine, loss_handler, results, viz_handler, routine_args,
                                     quit_on_bad_values=quit_on_bad_values)

                # Update the losses results
                routine_losses.update(**dict((k, v.item()) for k, v in loss_handler.items()))
                for k, v in routine_losses.items():
                    if k in losses:
                        losses[k] += v
                    else:
                        losses[k] = v
            update_dict_of_lists(results['losses'], **losses)

            for model_key in exp.MODEL_HANDLER:
                reg.clip(model_key)  # weight clipping
                reg.l1_decay(model_key)  # l1 weight decay

    except StopIteration:
        pass

    for routine_key, routine in exp.FINISH_TRAIN_ROUTINES.items():
        for u in range(num_updates_dict[routine_key]):
            loss_handler = LossHandler()
            perform_routine(routine_key, routine, loss_handler, results, viz_handler, routine_args,
                            quit_on_bad_values=quit_on_bad_values)

    results = summarize_results(results)

    return results


def test_epoch(epoch, viz_handler, return_std=False):
    for k, model in exp.MODEL_HANDLER.items():
        if k == 'extras':
            continue
        if isinstance(model, (tuple, list)):
            for net in model:
                net.eval()
        else:
            model.eval()

    data.DATA_HANDLER.reset(test=True, string='Evaluating (epoch {}): '.format(epoch))
    results = {'losses': dict((rk, []) for rk in exp.MODEL_HANDLER.keys())}
    routine_args = exp.ARGS['test_routines']

    viz_handler.ignore = False
    try:
        while True:
            # Iterate data
            data.DATA_HANDLER.next()

            # Loop through routines
            losses = {}
            for routine_key, routine in exp.TEST_ROUTINES.items():
                if routine_key == 'final':
                    continue
                routine_losses = {}
                loss_handler = LossHandler()
                perform_routine(routine_key, routine, loss_handler, results, viz_handler, routine_args)

                # Update the losses results
                routine_losses.update(**dict((k, v.item()) for k, v in loss_handler.items()))
                for k, v in routine_losses.items():
                    if k in losses:
                        losses[k] += v
                    else:
                        losses[k] = v
            update_dict_of_lists(results['losses'], **losses)
            viz_handler.ignore = True

    except StopIteration:
        pass

    for routine_key, routine in exp.FINISH_TEST_ROUTINES.items():
        loss_handler = LossHandler()
        perform_routine(routine_key, routine, loss_handler, results, viz_handler, routine_args)

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


def main_loop(epochs=None, archive_every=None, test_mode=False, quit_on_bad_values=False, save_on_best=None):
    info = pprint.pformat(exp.ARGS)
    viz.visualizer.text(info, env=exp.NAME, win='info')
    vh = VizHandler()
    total_time = 0.
    if test_mode:
        test_results, test_std = test_epoch('Testing', vh, return_std=True)
        logger.info(' | '.join(
            ['{}: {:.5f}({:.5f})'.format(k, test_results[k], test_std[k]) for k in test_results.keys()]))
        exit(0)
    best = None

    try:
        for e in range(epochs):
            epoch = exp.INFO['epoch']

            start_time = time.time()

            # TRAINING
            train_results_ = train_epoch(epoch, vh, quit_on_bad_values)
            convert_to_numpy(train_results_)
            update_dict_of_lists(exp.SUMMARY['train'], **train_results_)

            if save_on_best:
                flattened_results = {}
                for k, v in train_results_.items():
                    if isinstance(v, dict):
                        for k_, v_ in v.items():
                            flattened_results[k + '.' + k_] = v_
                    else:
                        flattened_results[k] = v
                if save_on_best in flattened_results:
                    # This needs to be fixed. when train_for is set, result keys vary per epoch
                    #if save_on_best not in flattened_results:
                    #    raise ValueError('`save_on_best` key `{}` not found. Available: {}'.format(
                    #        save_on_best, tuple(flattened_results.keys())))
                    current = flattened_results[save_on_best]
                    if not best or current > best:
                        best = current
                        print('\nFound best {} (train): {}'.format(save_on_best, best))
                        exp.save(prefix='best_' + save_on_best)

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

    logger.info('Successfully completed training')
    exp.save(prefix='final')
