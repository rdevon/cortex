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
from .utils import convert_to_numpy, update_dict_of_lists
from .viz import VizHandler, plot


logger = logging.getLogger('cortex.train')


def setup():
    # Test the routines and recover the loss keys
    with torch.no_grad():
        data.DATA_HANDLER.reset('train', make_pbar=False)
        models.MODEL.run_procedure(0)


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


def train_epoch(epoch, quit_on_bad_values, eval_during_train, data_mode='train'):
    models.MODEL.reset()
    models.MODEL.set_train()
    models.MODEL.viz.ignore = True
    data.DATA_HANDLER.reset(data_mode, string='Training (epoch {}): '.format(epoch))

    try:
        while True:
            models.MODEL.train(0, quit_on_bad_values=quit_on_bad_values)

            for net_key in models.NETWORK_HANDLER:
                reg.clip(net_key)  # weight clipping
                reg.l1_decay(net_key)  # l1 weight decay

    except StopIteration:
        pass

    if not eval_during_train:
        return test_epoch(epoch, data_mode=data_mode)

    results = summarize_results(models.MODEL.results)
    return results


def test_epoch(epoch, eval_mode=False, data_mode='test'):
    models.MODEL.reset()
    models.MODEL.set_eval()
    data.DATA_HANDLER.reset(data_mode, string='Evaluating (epoch {}): '.format(epoch))

    models.MODEL.viz.ignore = False
    try:
        while True:
            models.MODEL.run_procedure(0)
            models.MODEL.viz.ignore = True

    except StopIteration:
        pass

    means = summarize_results(models.MODEL.results)

    if eval_mode:
        if models.ARCH.eval_routine is not None:
            models.ARCH.eval_routine(data.DATA_HANDLER, models.MODEL_HANDLER, means, viz_handler)
        stds = summarize_results_std(results)
        return means, stds

    return means


def display_results(train_results, test_results, epoch, epochs, epoch_time, total_time):
    if epochs and epoch:
        print('\n\tEpoch {}/{} took {:.3f}s. Total time: {:.2f}'.format(epoch + 1, epochs, epoch_time, total_time))

    times = train_results.pop('time', None)
    if times:
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


def main_loop(epochs=500, archive_every=10, quit_on_bad_values=True, save_on_best=None,
              save_on_lowest=None, save_on_highest=None, eval_during_train=True,
              train_mode='train', test_mode='test', eval_only=False):
    '''

    Args:
        epochs: Number of epochs.
        archive_every: Number of epochs for writing checkpoints.
        quit_on_bad_values: Training data mode.
        save_on_best: Testing data mode.
        save_on_lowest: Test on data only (no training).
        save_on_highest: Quit when nans or infs found.
        eval_during_train: Saves when highest of this result is found.
        train_mode: Saves when highest of this result is found.
        test_mode: Saves when lowest of this result is found.
        eval_only: Gives results over a training epoch.

    '''
    info = pprint.pformat(exp.ARGS)
    viz.visualizer.text(info, env=exp.NAME, win='info')
    total_time = 0.
    if eval_only:
        test_results, test_std = test_epoch('Testing', vh, eval_mode=True, mode=test_mode)
        convert_to_numpy(test_results)
        convert_to_numpy(test_std)
        
        display_results(test_results, test_std, 'Evaluation', None, None, None)
        exit(0)
    best = None

    try:
        for e in range(epochs):
            epoch = exp.INFO['epoch']

            start_time = time.time()

            # TRAINING
            train_results_ = train_epoch(epoch, quit_on_bad_values, eval_during_train,
                                         data_mode=train_mode)
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
                    # TODO(Devon) This needs to be fixed. when train_for is set, result keys vary per epoch
                    # if save_on_best not in flattened_results:
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
            test_results_ = test_epoch(epoch, data_mode=test_mode)
            convert_to_numpy(test_results_)
            update_dict_of_lists(exp.SUMMARY['test'], **test_results_)
            align_summaries(exp.SUMMARY['train'], exp.SUMMARY['test'])

            # Finishing up
            epoch_time = time.time() - start_time
            total_time += epoch_time
            display_results(train_results_, test_results_, e, epochs, epoch_time, total_time)
            plot()
            models.MODEL.viz.show()
            models.MODEL.viz.clear()
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
