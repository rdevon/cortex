'''Module for training.

'''

__author__ = 'R Devon Hjelm'
__author_email__ = 'erroneus@gmail.com'

import logging
import pprint
import sys
import time

import numpy as np
import torch

from . import data, exp, models, optimizer, viz, reg
from .utils import bad_values, convert_to_numpy, Handler, update_dict_of_lists
from .viz import VizHandler, plot


logger = logging.getLogger('cortex.train')

ROUTINE_MODELS = {}

_args = dict(
    epochs=500,
    archive_every=10,
    test_mode=False,
    quit_on_bad_values=False,
    save_on_best=None,
    save_on_lowest=None,
    save_on_highest=None
)

_args_help = dict(
    epochs='Number of epochs',
    archive_every='Number of epochs for writing checkpoints.',
    test_mode='Testing mode. No training.',
    quit_on_bad_values='Quit when nans or infs found.',
    save_on_best='Saves when highest of this result is found.',
    save_on_highest='Saves when highest of this result is found.',
    save_on_lowest='Saves when lowest of this result is found.'
)


class LossHandler(Handler):
    '''
    Simple dict-like container for losses
    '''

    _type = torch.Tensor
    _get_error_string = 'Loss `{}` not found. You must add it as a dict entry'

    def check_key_value(self, k, v):
        super().check_key_value(k, v)
        if k not in models.MODEL_HANDLER:
            raise AttributeError('Keyword `{}` not in the model_handler. Found: {}.'.format(
                k, tuple(models.MODEL_HANDLER.keys())))

    def __setitem__(self, k, v):
        self.check_key_value(k, v)
        if len(v.size()) > 0:
            raise ValueError('Loss size must be a scalar. Got {}'.format(v.size()))
        super().__setitem__(k, v)


def setup():
    # Test the routines and recover the loss keys
    with torch.no_grad():
        data.DATA_HANDLER.reset(make_pbar=False)
        data.DATA_HANDLER.next()
        routine_models = {}
        args = exp.ARGS['routines']

        for routine_key, routine in models.ARCH.train_routines.items():
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
        optimizer.OPTIMIZERS[model_key].zero_grad()

    # Set requires grad from training models.
    for mk, model in models.MODEL_HANDLER.items():
        if isinstance(model, (list, tuple)):
            for net in model:
                for p in net.parameters():
                    p.requires_grad = mk in ROUTINE_MODELS[routine_key]
        else:
            for p in model.parameters():
                p.requires_grad = mk in ROUTINE_MODELS[routine_key]

    # Perform routine
    start_time = time.time()
    perform_routine(routine_key, routine, loss_handler, results, viz_handler, args, quit_on_bad_values=quit_on_bad_values)

    for model_key, loss in loss_handler.items():
        # Do backward step
        if loss is not None:
            loss.backward()
            optimizer.OPTIMIZERS[model_key].step()

    end_time = time.time()
    results['time'][routine_key].append(end_time - start_time)


def perform_routine(routine_key, routine, loss_handler, results, viz_handler, args, quit_on_bad_values=False):
    if routine is None:
        return

    if isinstance(args, dict) and routine_key in args:
        args = args[routine_key]

    # Run routine
    routine_results = {}
    if exp.DEVICE == torch.device('cpu'):
        routine(data.DATA_HANDLER, models.MODEL_HANDLER, loss_handler, routine_results, viz_handler, **args)
    else:
        with torch.cuda.device(exp.DEVICE.index):
            routine(data.DATA_HANDLER, models.MODEL_HANDLER, loss_handler, routine_results, viz_handler, **args)

    # Check for bad numbers
    bads = bad_values(routine_results)
    if bads and quit_on_bad_values:
        logger.error('Bad values found (quitting): {} \n All:{}'.format(bads, routine_results))
        exit(0)

    # Update results
    update_dict_of_lists(results, **routine_results)


def set_updates_dict(epoch):
    routines = {}
    routines.update(**models.ARCH.train_routines)
    routines.update(**models.ARCH.finish_train_routines)

    if optimizer.TRAIN_FOR is not None:
        total_steps = sum(optimizer.TRAIN_FOR.values())
        step = epoch % total_steps
        ts = 0
        for mk, s in optimizer.TRAIN_FOR.items():
            ts += s
            if step < ts:
                break
        num_updates_dict = dict((k, 0) for k in routines.keys())
        num_updates_dict[mk] = 1
    else:
        num_updates_dict = optimizer.UPDATES

    return num_updates_dict


def train_epoch(epoch, viz_handler, quit_on_bad_values):
    for k, model in models.MODEL_HANDLER.items():
        if isinstance(model, (tuple, list)):
            for net in model:
                net.train()
        else:
            model.train()

    results = {'time': dict((rk, []) for rk in models.ARCH.train_routines),
               'losses': dict((mk, []) for mk in models.MODEL_HANDLER.keys())}

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
            for routine_key, routine in models.ARCH.train_routines.items():
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

            for model_key in models.MODEL_HANDLER:
                reg.clip(model_key)  # weight clipping
                reg.l1_decay(model_key)  # l1 weight decay

    except StopIteration:
        pass

    for routine_key, routine in models.ARCH.finish_train_routines.items():
        for u in range(num_updates_dict[routine_key]):
            loss_handler = LossHandler()
            perform_routine(routine_key, routine, loss_handler, results, viz_handler, routine_args,
                            quit_on_bad_values=quit_on_bad_values)

    results = summarize_results(results)

    return results


def test_epoch(epoch, viz_handler, return_std=False):
    for k, model in models.MODEL_HANDLER.items():
        if k == 'extras':
            continue
        if isinstance(model, (tuple, list)):
            for net in model:
                net.eval()
        else:
            model.eval()

    data.DATA_HANDLER.reset(test=True, string='Evaluating (epoch {}): '.format(epoch))
    results = {'losses': dict((rk, []) for rk in models.MODEL_HANDLER.keys())}
    routine_args = exp.ARGS['test_routines']

    viz_handler.ignore = False
    try:
        while True:
            # Iterate data
            data.DATA_HANDLER.next()

            # Loop through routines
            losses = {}
            for routine_key, routine in models.ARCH.test_routines.items():
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

    for routine_key, routine in models.ARCH.finish_test_routines.items():
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


def main_loop(epochs=None, archive_every=None, test_mode=None, quit_on_bad_values=None, save_on_best=None,
              save_on_lowest=None, save_on_highest=None):
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

            if save_on_best or save_on_highest or save_on_lowest:
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
                    if not best:
                        found_best = True
                    elif save_on_lowest:
                        found_best = current < best
                    else:
                        found_best = current > best
                    if found_best:
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
