'''Module for training.

'''

from collections import OrderedDict
import logging
import pprint
import sys
import time

import numpy as np

from . import exp, viz
from .utils import convert_to_numpy, update_dict_of_lists
from .viz import plot

__author__ = 'R Devon Hjelm'
__author_email__ = 'erroneus@gmail.com'

logger = logging.getLogger('cortex.train')


def summarize_results(results):
    ''' Summarizes results from a dictionary of lists.

    Simply takes the mean of every list.

    Args:
        results (dict): Dictionary of list of results.

    Returns:
        dict: Dictionary of means.

    '''
    results_ = {}
    for k, v in results.items():
        if isinstance(v, dict):
            results_[k] = summarize_results(v)
        else:
            if len(v) > 0:
                try:
                    results_[k] = np.mean(v)
                except BaseException:
                    raise ValueError(
                        'Something is wrong with result {} of type {}.'.format(
                            k, type(
                                v[0])))

    return results_


def summarize_results_std(results):
    '''Standard deviation version of `summarize_results`

    Args:
        results (dict): Dictionary of list of results.

    Returns:
        dict: Dictionary of stds.

    '''
    results_ = {}
    for k, v in results.items():
        if isinstance(v, dict):
            results_[k] = summarize_results(v)
        else:
            results_[k] = np.std(v)
    return results_


def train_epoch(model, epoch, eval_during_train, data_mode='train',
                use_pbar=True):
    '''Trains model.

    Goes until data iterator returns StopIteration.

    Args:
        model (ModelPlugin): Model to train.
        epoch (int): Epoch
        eval_during_train (bool): Pull results during training.
            If false, recompute results keeping model fixed.
        data_mode (str): train or test
        use_pbar (bool): Use a progressbar when iterating data.

    Returns:
        dict: Dictionary of results for each batch.

    '''
    model.train_loop(epoch, data_mode=data_mode, use_pbar=use_pbar)

    if not eval_during_train:
        return test_epoch(model, epoch, data_mode=data_mode, use_pbar=use_pbar)

    results = summarize_results(model._epoch_results)
    return results


def test_epoch(model, epoch, data_mode='test', use_pbar=True):
    '''Evaluates model.

    Goes until data iterator returns StopIteration.

    Args:
        model (ModelPlugin): Model to train.
        epoch (int): Epoch
        data_mode (str): train or test
        use_pbar (bool): Use a progressbar when iterating data.

    Returns:
        dict: Dictionary of results for each batch.

    '''
    model.eval_loop(epoch, data_mode=data_mode, use_pbar=use_pbar)
    results = summarize_results(model._epoch_results)

    model.data.reset(make_pbar=False, mode='test')
    model.data.next()
    model.clear_viz()
    model.visualize(auto_input=True)
    return results


def display_results(train_results, test_results, last_train_results, last_test_results,
                    epoch, epochs, epoch_time, total_time):
    '''

    Args:
        train_results (dict): Dictionary of results from training data.
        test_results (dict): Dictionary of results from holdout data.
        last_train_results (dict or None): Dictionary of last training data results.
        last_test_results (dict or None: Dictionary of last holdout data results.
        epoch (int): Current epoch.
        epochs (int): Total number of epochs.
        epoch_time (float): Time for this epoch.
        total_time (float): Total time for training.

    '''

    class bcolors:
        INCREASING = '\033[91m' # red
        DECREASING ='\033[94m' # blue
        UNDERLINE = '\033[4m'
        BOLD = '\033[1m'
        ENDC = '\033[0m'

    def color_increasing(s):
        return bcolors.INCREASING + s[:-1] + '\u21e7' + bcolors.ENDC

    def color_decreasing(s):
        return bcolors.DECREASING + s[:-1] + '\u21e9' + bcolors.ENDC

    def underline(s):
        return bcolors.UNDERLINE + s + bcolors.ENDC

    def bold(s):
        return bcolors.BOLD + s + bcolors.ENDC

    format_length = 8
    format_string = '{:8.4f}'

    def print_table(train, test, train_last, test_last, prefix):
        '''Prints table.

        Args:
            train (dict): Dictionary of results from training data.
            test (dict): Dictionary of results from holdout data.
            train_last (dict or None): Dictionary of last training data results.
            test_last (dict or None): Dictionary of last holdout data results.
            prefix: Prefix for first line of table.

        '''
        train = OrderedDict(sorted(train.items()))
        max_key_length = max(len(k) for k in train.keys())
        s = prefix + \
            ' ' * (max_key_length + 2 - len(prefix) + 4 + format_length - len('train')) + \
            ' Train |' + \
            ' ' * (format_length - len('test')) + ' Test'
        s = underline(s)
        print(s)
        for k in train.keys():
            if k in ('losses', 'times'):
                continue
            s = '    '
            key_length = len(k)
            v_train = train[k]
            v_test = test[k]
            v_train_last = train_last[k] if (train_last and k in train_last) else v_train
            v_test_last = test_last[k] if (test_last and k in test_last) else v_test

            s += '{} {} '.format(k, '-' + '-' * (max_key_length - key_length))
            s_train = format_string.format(v_train)
            if v_train != v_train_last:
                if v_train > v_train_last:
                    s += color_increasing(s_train)
                elif v_train < v_train_last:
                    s += color_decreasing(s_train)
            else:
                s += s_train

            if v_test is not None:
                s_test = format_string.format(v_test)
                if v_test != v_test_last:
                    s += ' | '
                    if v_test > v_test_last:
                        s += color_increasing(s_test)
                    elif v_test < v_test_last:
                        s += color_decreasing(s_test)
                else:
                    s += ' | ' + s_test

            print(s)

    print()
    print()
    # Show epoch information
    if epochs and epoch:
        s = 'Epoch {} / {} took {:.2f}s. Total time: {:.2f}s'\
            .format(epoch + 1, epochs, epoch_time, total_time)
        s = bold(s)
        print(s)

    # Show times
    times = train_results.pop('times', None)
    print(underline('Avg update times: '))
    if times:
        for k, v in times.items():
            print('    {}: {:.5f}s'.format(k, v))

    # Show losses
    train_losses = train_results['losses']
    test_losses = test_results['losses']
    train_losses_last = last_train_results.pop('losses') if last_train_results else None
    test_losses_last = last_test_results.pop('losses') if last_test_results else None
    print()
    print_table(train_losses, test_losses, train_losses_last, test_losses_last, 'Network losses:')
    print()
    print_table(train_results, test_results, last_train_results, last_test_results, 'Results:')
    print()


def align_summaries(d_train, d_test):
    '''Aligns summaries for models that are updated at different rates.

    Args:
        d_train: Dictionary of results from training data.
        d_test: Dictionary of results from holdout data.

    '''
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
                        v_train[k_] = (v_train[k_] + [v_train[k_][-1]] *
                                       (max_len - len(v_train[k_])))
                for k_, v in v_test.items():
                    if len(v) < max_len:
                        v_test[k_] = (v_test[k_] + [v_test[k_][-1]] *
                                      (max_len - len(v_test[k_])))
            else:
                if len(v_train) > len(v_test):
                    d_test[k] = (v_test + [v_test[-1]] *
                                 (len(v_train) - len(v_test)))
                elif len(v_test) > len(v_train):
                    d_train[k] = (v_train + [v_train[-1]] *
                                  (len(v_test) - len(v_train)))
        elif k in d_train:
            v_train = d_train[k]
            if isinstance(v_train, dict):
                max_len = max([len(v) for v in v_train.values()])
                for k_, v in v_train.items():
                    if len(v) < max_len:
                        v_train[k_] = (v_train[k_] + [v_train[k_][-1]] *
                                       (max_len - len(v_train[k_])))
        elif k in d_test:
            v_test = d_test[k]
            if isinstance(v_test, dict):
                max_len = max([len(v) for v in v_test.values()])
                for k_, v in v_test.items():
                    if len(v) < max_len:
                        v_test[k_] = v_test[k_] + [v_test[k_][-1]] * \
                            (max_len - len(v_test[k_]))


def save_best(model, train_results, best, save_on_best, save_on_lowest):
    '''Saves the best model according to some metric.

    Args:
        model (ModelPlugin): Model.
        train_results (dict): Dictionary of results from training data.
        best (float): Last best value.
        save_on_best (bool): If true, save when best is found.
        save_on_lowest (bool): If true, lower is better.

    Returns:
        float: the best value.

    '''
    flattened_results = {}
    for k, v in train_results.items():
        if isinstance(v, dict):
            for k_, v_ in v.items():
                flattened_results[k + '.' + k_] = v_
        else:
            flattened_results[k] = v
    if save_on_best in flattened_results:
        # TODO(Devon) This needs to be fixed.
        # when train_for is set, result keys vary per epoch
        # if save_on_best not in flattened_results:
        #    raise ValueError('`save_on_best` key `{}` not found.
        #  Available: {}'.format(
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
            print(
                '\nFound best {} (train): {}'.format(
                    save_on_best, best))
            exp.save(model, prefix='best_' + save_on_best)

        return best


def main_loop(model, epochs=500, archive_every=10, save_on_best=None,
              save_on_lowest=None, save_on_highest=None, eval_during_train=True,
              train_mode='train', test_mode='test', eval_only=False,
              pbar_off=False):
    '''

    Args:
        epochs: Number of epochs.
        archive_every: Number of epochs for writing checkpoints.
        save_on_best: Testing data mode.
        save_on_lowest: Test on data only (no training).
        save_on_highest: Quit when nans or infs found.
        eval_during_train: Saves when highest of this result is found.
        train_mode: Saves when highest of this result is found.
        test_mode: Saves when lowest of this result is found.
        eval_only: Gives results over a training epoch.
        pbar_off: Turn off the progressbar.

    '''
    info = pprint.pformat(exp.ARGS)

    logger.info('Starting main loop.')

    if (viz.visualizer):
        viz.visualizer.text(info, env=exp.NAME, win='info')
    total_time = 0.
    if eval_only:
        test_results, test_std = test_epoch(
            'Testing', eval_mode=True, mode=test_mode)
        convert_to_numpy(test_results)
        convert_to_numpy(test_std)

        display_results(test_results, test_std, 'Evaluation', None, None, None)
        exit(0)
    best = None
    if not isinstance(epochs, int):
            epochs = epochs['epochs']

    epoch = exp.INFO['epoch']
    first_epoch = epoch
    train_results_last_ = None
    test_results_last_ = None

    while epoch < epochs:
        try:
            epoch = exp.INFO['epoch']
            logger.info('Epoch {} / {}'.format(epoch, epochs))
            start_time = time.time()

            # TRAINING
            train_results_ = train_epoch(
                model, epoch, eval_during_train,
                data_mode=train_mode, use_pbar=not(pbar_off))
            convert_to_numpy(train_results_)
            update_dict_of_lists(exp.SUMMARY['train'], **train_results_)

            if save_on_best or save_on_highest or save_on_lowest:
                best = save_best(model, train_results_, best, save_on_best,
                                 save_on_lowest)

            # TESTING
            test_results_ = test_epoch(model, epoch, data_mode=test_mode,
                                       use_pbar=not(pbar_off))

            convert_to_numpy(test_results_)
            update_dict_of_lists(exp.SUMMARY['test'], **test_results_)
            align_summaries(exp.SUMMARY['train'], exp.SUMMARY['test'])

            # Finishing up
            epoch_time = time.time() - start_time
            total_time += epoch_time
            display_results(train_results_, test_results_, train_results_last_, test_results_last_,
                            epoch, epochs, epoch_time, total_time)

            train_results_last_ = train_results_
            test_results_last_ = test_results_

            if viz.visualizer:
                plot(epoch, init=(epoch == first_epoch))
                model.viz.show()
                model.viz.clear()

            if (archive_every and epoch % archive_every == 0):
                exp.save(model, prefix=epoch)
            else:
                exp.save(model, prefix='last')

            exp.INFO['epoch'] += 1

        except KeyboardInterrupt:
            def stop_training_query():
                while True:
                    try:
                        response = input('Keyboard interrupt. Kill? (Y/N) '
                                         '(or ^c again)')
                    except KeyboardInterrupt:
                        return True
                    response = response.lower()
                    if response == 'y':
                        return True
                    elif response == 'n':
                        print('Cancelling interrupt. Starting epoch over.')
                        return False
                    else:
                        print('Unknown response')

            kill = stop_training_query()

            if kill:
                print('Training interrupted')
                exp.save(model, prefix='interrupted')
                sys.exit(0)

    logger.info('Successfully completed training')
    exp.save(model, prefix='final')
