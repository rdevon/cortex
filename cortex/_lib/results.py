'''Module for functions for handling results.

'''

from collections import OrderedDict
import numpy as np

from cortex._lib import exp
from cortex._lib.utils import convert_to_numpy, update_dict_of_lists


class bcolors:
    INCREASING = '\033[91m'  # red
    DECREASING = '\033[94m'  # blue
    UNDERLINE = '\033[4m'
    BOLD = '\033[1m'
    ENDC = '\033[0m'


def color_increasing(s, no_ascii=False):
    if not no_ascii:
        return bcolors.INCREASING + s[:-1] + '\u21e7' + bcolors.ENDC
    else:
        return s[:-1] + '+'


def color_decreasing(s, no_ascii=False):
    if not no_ascii:
        return bcolors.DECREASING + s[:-1] + '\u21e9' + bcolors.ENDC
    else:
        return s[:-1] + '-'


def underline(s, no_ascii=False):
    if not no_ascii:
        return bcolors.UNDERLINE + s + bcolors.ENDC
    else:
        return s


def summarize_results(results, start, end):
    ''' Summarizes results from a dictionary of lists.

    Simply takes the mean of every list.

    Args:
        results (dict): Dictionary of list of results.
        start (int): Start of summary window.
        end (int): End of summary window.

    Returns:
        dict: Dictionary of means or list of means.

    '''

    def chunks(l, n):
        """Yield successive n-sized chunks from l."""
        for i in range(0, len(l), n):
            yield np.mean(l[i:i + n])

    results_ = {}
    for k, v in results.items():
        if isinstance(v, dict):
            results_[k] = summarize_results(v, length=length)
        else:
            if len(v) > 0:
                try:
                    results_[k] = list(chunks(v, length))
                except BaseException:
                    raise ValueError(
                        'Something is wrong with result {} of type {}.'.format(
                            k, type(v[0])))

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
                if len(v_train) == 0:
                    max_train_len = 0
                else:
                    max_train_len = max([len(v) for v in v_train.values()])
                if len(v_test) == 0:
                    max_test_len = 0
                else:
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


class Results(object):
    '''Class for handling results.

    '''

    def __init__(self):
        # Results for an epoch.

        # Results are a triple: the results themselves, the update step the result was given and the epoch.
        self.results = dict(losses=dict(), results=dict(), times=dict(), grads=dict())
        # Epoch number for every result.
        self.epoch_markers = dict(losses=dict(), results=dict(), times=dict(), grads=dict())
        # Data step number for every result.
        self.step_markers = dict(losses=dict(), results=dict(), times=dict(), grads=dict())

        self._pause = False

    def clear(self):
        self.results.clear()
        self.epoch_markers.clear()
        self.step_markers.clear()

    def todict(self):
        d = dict(results=self.results,
                 epoch_markers=self.epoch_markers,
                 step_markers=self.step_markers)
        return d

    def load(self, results, epoch_markers, step_markers):
        self.results = results
        self.epoch_markers = epoch_markers
        self.step_markers = step_markers

    def pause(self):
        '''Handler will ignore results.

        '''
        self._pause = True

    def unpause(self):
        self._pause = False

    def log_result(self, mode, results, new_results, data_steps, epochs, collapse=True):
        '''Logs the result in addition to the epoch and data step the result was logged.

        '''

        if mode not in results:
            results[mode] = dict()
            data_steps[mode] = dict()
            epochs[mode] = dict()

        convert_to_numpy(new_results)

        if collapse:
            new_results = self.collapse_result(new_results)

        for key, value in new_results.items():
            if key not in results[mode]:
                results[mode][key] = [value]
                data_steps[mode][key] = [exp.INFO['data_steps']]
                epochs[mode][key] = [exp.INFO['epoch']]
            else:
                results[mode][key].append(value)
                data_steps[mode][key].append(exp.INFO['data_steps'])
                epochs[mode][key].append(exp.INFO['epoch'])

    def update(self, mode, results=None, times=None, grads=None):
        '''

        Args:
            mode (str): Data mode.
            results (dict): Main results.
            times (dict): Times for routines.
            grads (dict): Gradients

        '''

        if self._pause:
            return

        for result_type in 'results', 'times', 'grads':
            d = self.results[result_type]
            e = self.epoch_markers[result_type]
            s = self.step_markers[result_type]

            if result_type == 'results':
                result = results
            elif result_type == 'times':
                result = times
            elif result_type == 'grads':
                result = grads

            if result is not None:
                self.log_result(mode, d, result, s, e)

    def update_losses(self, losses, mode):
        '''Added losses to the results.

        Args:
            losses (dict): Loss results.
            mode (str): Data mode.

        '''

        if self._pause:
            return

        def collapse(d, losses, model_name=None):
            for k, v in d.items():
                if isinstance(v, dict):
                    collapse(v, losses, model_name=k)
                else:
                    v_ = v.detach().item()
                    if model_name is not None:
                        if k not in losses:
                            losses[k] = {}
                        losses[k][model_name] = v_
                        if 'all' in losses[k]:
                            losses[k]['all'] += v_
                        else:
                            losses[k]['all'] = v_

        def summarize(losses):
            d = {}
            for k, v in losses.items():
                d[k] = v.pop('all')
                if len(v) > 1:  # more than one loss for this network
                    for k_, v_ in v.items():
                        d['{}.{}'.format(k, k_)] = v_
            return d

        loss_summary = {}
        collapse(losses, loss_summary)
        loss_summary = summarize(loss_summary)

        d = self.results['losses']
        e = self.epoch_markers['losses']
        s = self.step_markers['losses']

        self.log_result(mode, d, loss_summary, s, e, collapse=False)

    def collapse_result(self, results):
        '''Collapses results from dictionary.

        '''
        def collapse(d, collapsed_results, prefix=''):
            for k, v in d.items():
                if prefix == '':
                    key = k
                else:
                    key = prefix + '.' + k
                if isinstance(v, dict):
                    collapse(v, collapsed_results, prefix=key)
                else:
                    collapsed_results[key] = v
        collapsed_results = {}
        collapse(results, collapsed_results)
        return collapsed_results

    def pull(self, mode, result_type, result_key, epoch, average=True):
        '''Pulls a result from an epoch.

        Args:
            mode (str): Data mode.
            result_type (str): Result type (results, times, losses, or grads)
            result_key (str): Result name.
            epoch (int): Epoch to pull result from.
            average (bool): Average results from each epoch.

        Returns:
            float or list: Result

        '''

        if mode in self.results[result_type] and result_key in self.results[result_type][mode]:
            d = self.results[result_type][mode][result_key]
        else:
            return {}
        s = self.epoch_markers[result_type][mode][result_key]

        idx = [i for i in range(len(s)) if s[i] == epoch]
        if len(idx) == 0:
            return None
        else:
            result = [d[i] for i in idx]
            if average:
                result = np.mean(result)

        return result

    def pull_all(self, mode, result_type, epoch, average=True):
        '''Pulls results from an epoch.

        Args:
            mode (str): Data mode.
            result_type (str): Result type (results, times, losses, or grads)
            epoch (int): Epoch to pull result from.
            average (bool): Average results from each epoch.

        Returns:
            dict: Results
        '''
        if mode in self.results[result_type]:
            d = self.results[result_type][mode]
        else:
            return {}

        pulled_results = dict()
        for k in d.keys():
            pulled_results[k] = self.pull(mode, result_type, k, epoch, average=average)
        return pulled_results

    def chunk(self, mode, result_type, result_key, start=0, end=None, window=0):
        '''Chunks a result across steps.

        Args:
            mode (str): Data mode.
            result_type (str): Result type (results, times, losses, or grads)
            result_key (str): Result name.
            start (int): Start of chunking.
            end (int): End of chunking
            window (int): Window size for chunks. If 0, then chunk by epoch.

        Returns:
            list: Chunked results
        '''

        if mode in self.results[result_type] and result_key in self.results[result_type][mode]:
            Y = self.results[result_type][mode][result_key]
        else:
            return (None, None)

        if window == 0:
            X = self.epoch_markers[result_type][mode][result_key]
        else:
            X = self.step_markers[result_type][mode][result_key]

        def chunks(l, start, end, window, average=True):
            if start == 0:
                yield l[0]
                l = l[1:]
                end -= 1

            for i in range(start, end, window):
                if average:
                    yield np.mean(l[i:i+window])
                else:
                    try:
                        yield l[i+window-1]
                    except IndexError:
                        yield l[-1]

        if window == 0:
            # Chunking by epoch
            end = end or len(X)

            # Visualization is usually triggered after one step in new epoch
            # Should be fixable in viz
            if mode == 'train':
                max_X = max(X)
            else:
                max_X = max(X) + 1

            epochs = list(range(start, max_X))
            if len(X) == 0:
                Y = None
            else:
                Y_new = []
                for epoch in epochs:
                    idx = [i for i in range(start, end) if X[i] == epoch]
                    Y_epoch = np.mean([Y[i] for i in idx])
                    Y_new.append(Y_epoch)
                Y = Y_new

            X = epochs

        else:
            # Chunking by update window
            if mode == 'train':
                end = end or len(X)
                X = list(chunks(X, start, end, window, average=False))
                Y = list(chunks(Y, start, end, window))
            elif mode == 'test':
                X_unique = list(np.unique(X))
                X_unique = [X for X in X_unique if X >= start]
                if len(X_unique) == 0:
                    return ([], [])
                new_Y = []
                for idx in X_unique:
                    X_mask = [idx == i for i in X]
                    new_Y.append(sum([Y[i] * X_mask[i] for i in range(len(X_mask))]) / float(sum(X_mask)))

                Y = new_Y
                end = end or max(X_unique)
                windows = list(range(start, end + window, window))
                X = []
                new_Y = []

                i = 0
                for idx in windows:
                    Y_ = []
                    while (i < len(X_unique)) and (X_unique[i] <= idx):
                        Y_.append(Y[i])
                        i += 1
                    if len(Y_) != 0:
                        new_Y.append(np.mean(Y_))
                        X.append(idx)
                Y = new_Y

        return X, Y

    def chunk_all(self, mode, result_type, start=0, end=None, window=0):
        '''Chunks results across steps.

        Args:
            mode (str): Data mode.
            result_type (str): Result type (results, times, losses, or grads)
            start (int): Start of chunking.
            end (int): End of chunking
            window (int): Window size for chunks. If 0, then chunk by epoch.

        Returns:
            dict: All chunked results
        '''
        if mode in self.results[result_type]:
            d = self.results[result_type][mode]
        else:
            return (None, None)

        chunked_results = dict()
        chunked_indices = dict()
        for k in d.keys():
            X, Y = self.chunk(mode, result_type, k, start=start, end=end, window=window)

            chunked_indices[k] = X
            chunked_results[k] = Y

        return chunked_indices, chunked_results

    def print_table(self, train, test, train_last, test_last, prefix, no_ascii=False):
        '''Prints table.

        Args:
            train (dict): Dictionary of results from training data.
            test (dict): Dictionary of results from holdout data.
            train_last (dict or None): Dictionary of last training data results.
            test_last (dict or None): Dictionary of last holdout data results.
            prefix: Prefix for first line of table.

        '''

        format_length = 8
        format_string = '{:8.4f}'

        train = OrderedDict(sorted(train.items()))

        if len(train) == 0:
            return

        max_key_length = max(len(k) for k in train.keys())
        s = prefix + \
            ' ' * (max_key_length + 2 - len(prefix) + 4 + format_length - len('train')) + \
            ' Train |' + \
            ' ' * (format_length - len('test')) + ' Test'
        s = underline(s, no_ascii=no_ascii)

        print(s)
        for k in train.keys():
            s = '    '
            key_length = len(k)
            v_train = train[k]
            v_test = test.get(k, None)
            v_train_last = train_last[k] if (train_last and k in train_last) else v_train
            v_test_last = test_last[k] if (test_last and k in test_last) else v_test

            s += '{} {} '.format(k, '-' + '-' * (max_key_length - key_length))
            s_train = format_string.format(v_train)
            if v_train_last and v_train != v_train_last:
                if v_train > v_train_last:
                    s += color_increasing(s_train, no_ascii=no_ascii)
                elif v_train < v_train_last:
                    s += color_decreasing(s_train, no_ascii=no_ascii)
            else:
                s += s_train

            if v_test is not None:
                s_test = format_string.format(v_test)
                if v_test_last and v_test != v_test_last:
                    s += ' | '
                    if v_test > v_test_last:
                        s += color_increasing(s_test, no_ascii=no_ascii)
                    elif v_test < v_test_last:
                        s += color_decreasing(s_test, no_ascii=no_ascii)
                else:
                    s += ' | ' + s_test
            else:
                s += ' | N/A'

            print(s)
        print()

    def display(self, epoch=None, no_ascii=False):
        '''Shows results on the command line.

        Args:
            epoch (int or None): Epoch to display.
            no_ascii (bool): If True, do not display ascii characters or color.

        '''

        epoch = epoch or exp.INFO['epoch']
        print()

        for result_type in ('times', 'losses', 'results', 'grads'):
            train_results = self.pull_all(mode='train', result_type=result_type, epoch=epoch)

            if result_type == 'times' and train_results is not None:
                # Show times
                times = train_results

                print(underline('Avg update times: ', no_ascii=no_ascii))
                time_keys = sorted(list(times.keys()))
                for k in time_keys:
                    v = times[k]
                    print('    {}: {:.5f}s'.format(k, v))
                print()
            else:
                test_results = self.pull_all(mode='test', result_type=result_type, epoch=epoch)
                last_train_results = self.pull_all(mode='train', result_type=result_type, epoch=epoch - 1)
                last_test_results = self.pull_all(mode='test', result_type=result_type, epoch=epoch - 1)

                if result_type == 'losses':
                    header = 'Network losses:'
                elif result_type == 'results':
                    header = 'Results:'
                else:
                    header = 'Network grads:'

                self.print_table(train_results, test_results, last_train_results, last_test_results, header)