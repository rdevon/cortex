'''Module for training.

'''

import logging
from os import path
import pprint
import sys
import time

import numpy as np
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn

from .data import DATA_HANDLER
from . import exp, viz
from .utils import bad_values, compute_tsne, convert_to_numpy, update_dict_of_lists


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


def plot():
    train_summary = exp.SUMMARY['train']
    test_summary = exp.SUMMARY['test']
    for k in train_summary.keys():
        v_tr = np.array(train_summary[k])
        v_te = np.array(test_summary[k]) if k in test_summary.keys() else None
        if len(v_tr.shape) > 1:
            continue
        if v_te is not None:
            opts = dict(
                xlabel='updates',
                legend=['train', 'test'],
                ylabel=k,
                title=k)
            Y = np.column_stack((v_tr, v_te))
            X = np.column_stack((np.arange(v_tr.shape[0]), np.arange(v_tr.shape[0])))
        else:
            opts = dict(
                xlabel='updates',
                ylabel=k,
                title=k)
            Y = v_tr
            X = np.arange(v_tr.shape[0])
        viz.visualizer.line(Y=Y, X=X, env=exp.NAME, opts=opts, win='line_{}'.format(k))


def show(samples, prefix=''):
    prefix = exp.file_string(prefix)
    image_dir = exp.OUT_DIRS.get('image_dir', None)

    images = samples.get('images', {})
    for i, (k, v) in enumerate(images.items()):
        v = convert_to_numpy(v)
        logger.debug('Saving images to {}'.format(image_dir))
        if image_dir is None:
            out_path = path.join(image_dir, '{}_{}_samples.png'.format(prefix, k))
        else:
            out_path = None

        viz.save_images(v, 8, 8, out_file=out_path, labels=None, max_samples=64, image_id=1 + i, caption=k)

    scatters = samples.get('scatters', {})
    for i, (k, v) in enumerate(scatters.items()):
        if isinstance(v, tuple):
            v, l = v
        else:
            l = None
        v = convert_to_numpy(v)
        l = convert_to_numpy(l)

        if v.shape[1] == 1:
            raise ValueError('1D-scatter not supported')
        elif v.shape[1] > 2:
            logger.info('Scatter greater than 2D. Performing TSNE to 2D')
            v = compute_tsne(v)

        logger.debug('Saving scatter to {}'.format(image_dir))
        if image_dir is None:
            out_path = path.join(image_dir, '{}_{}_samples.png'.format(prefix, k))
        else:
            out_path = None

        viz.save_scatter(v, out_file=out_path, labels=l, image_id=i, title=k)

    histograms = samples.get('histograms', {})
    for i, (k, v) in enumerate(histograms.items()):
        convert_to_numpy(v)
        logger.debug('Saving histograms to {}'.format(image_dir))
        if image_dir is None:
            out_path = path.join(image_dir, '{}_{}_samples.png'.format(prefix, k))
        else:
            out_path = None
        viz.save_hist(v, out_file=out_path, hist_id=i)

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

    for k, model in exp.MODELS.items():
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


def train_epoch(epoch, quit_on_bad_values):
    for k, model in exp.MODELS.items():
        if isinstance(model, (tuple, list)):
            for net in model:
                net.train()
        else:
            model.train()

    DATA_HANDLER.reset(string='Training (epoch {}): '.format(epoch))

    results = {}

    try:
        while True:
            for i, k_ in enumerate(exp.MODELS.keys()):
                for _ in range(UPDATES[k_]):
                    DATA_HANDLER.next()

                    for k__, model in exp.MODELS.items():
                        if isinstance(model, (list, tuple)):
                            for net in model:
                                for p in net.parameters():
                                    p.requires_grad = (k__ == k_)
                        else:
                            for p in model.parameters():
                                p.requires_grad = (k__ == k_)

                    OPTIMIZERS[k_].zero_grad()

                    for k, v in exp.PROCEDURES.items():
                        start_time = time.time()
                        if k == 'main' or not isinstance(exp.ARGS['procedures'], dict):
                            args = exp.ARGS['procedures']
                        else:
                            args = exp.ARGS['procedures'][k]
                        losses, results_, _, _ = v(exp.MODELS, DATA_HANDLER, **args)
                        bads = bad_values(results_)
                        if bads and quit_on_bad_values:
                            logger.error('Bad values found (quitting): {} \n All:{}'.format(
                                bads, results_))
                            exit(0)

                        if isinstance(losses, dict):
                            if k_ in losses:
                                loss = losses[k_]
                            else:
                                loss = None
                        else:
                            loss = losses

                        if loss is not None:
                            loss.backward()
                        end_time = time.time()
                        results_['{}_{}_time'.format(k_, k)] = end_time - start_time
                        update_dict_of_lists(results, **results_)

                    OPTIMIZERS[k_].step()

                    if k_ in CLIPPING.keys():
                        clip = CLIPPING[k_]
                        model = exp.MODELS[k_]
                        if isinstance(model, (list, tuple)):
                            for net in model:
                                for p in net.parameters():
                                    p.data.clamp_(-clip, clip)
                        else:
                            for p in model.parameters():
                                p.data.clamp_(-clip, clip)

            '''
            tens = [obj for obj in gc.get_objects()
                if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data))]
            print(len(tens))
            for ten in tens:
                del ten
            '''
    except StopIteration:
        pass

    results = dict((k, np.mean(v)) for k, v in results.items())
    return results


def test_epoch(epoch, best_condition=0, return_std=False):
    for k, model in exp.MODELS.items():
        if isinstance(model, (tuple, list)):
            for net in model:
                net.eval()
        else:
            model.eval()

    DATA_HANDLER.reset(test=True, string='Evaluating (epoch {}): '.format(epoch))
    results = {}
    samples_ = None

    procedures = exp.ARGS['test_procedures']

    try:
        while True:
            DATA_HANDLER.next()
            samples__ = {}
            for k, v in exp.PROCEDURES.items():
                if k == 'main' or not isinstance(procedures, dict):
                    args = procedures
                else:
                    args = procedures[k]
                loss, results_, samples, condition = v(exp.MODELS, DATA_HANDLER, **args)
                if not samples_ and samples:
                    samples__.update(**samples)
                update_dict_of_lists(results, **results_)
            samples_ = samples_ or samples__
    except StopIteration:
        pass

    means = dict((k, np.mean(v)) for k, v in results.items())
    if return_std:
        stds = dict((k, np.std(v)) for k, v in results.items())
        return means, stds, samples_

    return means, samples_


def main_loop(summary_updates=None, epochs=None, updates_per_model=None, archive_every=None, test_mode=False,
              quit_on_bad_values=False):
    info = pprint.pformat(exp.ARGS)
    viz.visualizer.text(info, env=exp.NAME, win='info')
    if test_mode:
        test_results, test_std, samples_ = test_epoch('Testing', return_std=True)
        logger.info(' | '.join(
            ['{}: {:.5f}({:.5f})'.format(k, test_results[k], test_std[k]) for k in test_results.keys()]))
        exit(0)

    try:
        for e in range(epochs):
            epoch = exp.INFO['epoch']

            start_time = time.time()
            train_results_ = train_epoch(epoch, quit_on_bad_values)
            convert_to_numpy(train_results_)
            update_dict_of_lists(exp.SUMMARY['train'], **train_results_)

            test_results_, samples_ = test_epoch(epoch)
            convert_to_numpy(test_results_)
            update_dict_of_lists(exp.SUMMARY['test'], **test_results_)
            logger.info(' | '.join(['{}: {:.2f}/{:.2f}'.format(k, train_results_[k], test_results_[k] if k in test_results_.keys() else 0)
                                    for k in train_results_.keys()]))
            logger.info('Total Epoch {} of {} took {:.3f}s'.format(epoch + 1, epochs, time.time() - start_time))
            plot()
            show(samples_)
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
