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

from data import make_iterator
import exp
from utils import bad_values, update_dict_of_lists
import viz


logger = logging.getLogger('cortex.util')

OPTIMIZERS = {}


optimizer_defaults = dict(
    SGD=dict(momentum=0.9, weight_decay=5e-4),
    Adam=dict(betas=(0.5, 0.999))
)


def plot():
    train_summary = exp.SUMMARY['train']
    test_summary = exp.SUMMARY['test']
    for k in train_summary.keys():
        v_tr = np.array(train_summary[k])
        v_te = np.array(test_summary[k])
        opts = dict(
            xlabel='updates',
            legend=['train', 'test'],
            ylabel=k,
            title=k)
        if len(v_tr.shape) > 1:
            continue
        Y = np.column_stack((v_tr, v_te))
        X = np.column_stack((np.arange(v_tr.shape[0]), np.arange(v_tr.shape[0])))
        viz.visualizer.line(Y=Y, X=X, env=exp.NAME, opts=opts, win='line_{}'.format(k))


def show(samples, prefix=''):
    prefix = exp.file_string(prefix)
    image_dir = exp.OUT_DIRS.get('image_dir', None)

    images = samples.get('images', {})
    for i, (k, v) in enumerate(images.items()):
        logger.debug('Saving images to {}'.format(image_dir))
        if image_dir is None:
            out_path = path.join(image_dir, '{}_{}_samples.png'.format(prefix, k))
        else:
            out_path = None

        viz.save_images(v.cpu().numpy(), 8, 8, out_file=out_path, labels=None, max_samples=64, image_id=1 + i, caption=k)

    scatters = samples.get('scatters', {})
    for i, (k, v) in enumerate(scatters.items()):
        if isinstance(v, tuple):
            v, l = v
            l = l.cpu().numpy()
        else:
            l = None
        logger.debug('Saving scatter to {}'.format(image_dir))
        if image_dir is None:
            out_path = path.join(image_dir, '{}_{}_samples.png'.format(prefix, k))
        else:
            out_path = None

        viz.save_scatter(v.cpu().numpy(), out_file=out_path, labels=l, image_id=i, title=k)

def setup(optimizer=None, learning_rate=None, lr_decay=None, min_lr=None, decay_at_epoch=None,
          optimizer_options='default'):

    if optimizer_options == 'default' and optimizer in optimizer_defaults.keys():
        optimizer_options = optimizer_defaults[optimizer]

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
                model_params += net.parameters()
        else:
            if exp.USE_CUDA:
                model.cuda()
                model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
            model_params = model.parameters()

        logger.info('Training with {} and optimizer options {}'.format(optimizer, optimizer_options))
        if isinstance(learning_rate, dict):
            eta = learning_rate[k]
        else:
            eta = learning_rate

        optimizer = op(model_params, lr=eta, **optimizer_options)
        OPTIMIZERS[k] = optimizer

    if exp.USE_CUDA:
        cudnn.benchmark = True


def train_epoch(epoch):
    for k, model in exp.MODELS.items():
        if isinstance(model, (tuple, list)):
            for net in model:
                net.train()
        else:
            model.train()

    train_iter = make_iterator(string='Training (epoch {}): '.format(epoch))
    results = {}

    for inputs in train_iter:
        for i, k_ in enumerate(exp.MODELS.keys()):
            OPTIMIZERS[k_].zero_grad()

            for k, v in exp.PROCEDURES.items():
                if k == 'main' or not isinstance(exp.ARGS['procedures'], dict):
                    args = exp.ARGS['procedures']
                else:
                    args = exp.ARGS['procedures'][k]
                losses, results_, _, _ = v(exp.MODELS, inputs, **args)
                bads = bad_values(results_)
                if bads:
                    logger.error('Bad values found (quitting): {} \n All:{}'.format(
                        bads, results_))
                    exit(0)
                update_dict_of_lists(results, **results_)

                if isinstance(losses, dict):
                    if k_ in losses:
                        loss = losses[k_]
                    else:
                        loss = None
                else:
                    loss = losses

                if loss is not None:
                    loss.backward()

            OPTIMIZERS[k_].step()

    results = dict((k, np.mean(v)) for k, v in results.items())
    return results


def test_epoch(epoch, best_condition=0):
    for k, model in exp.MODELS.items():
        if isinstance(model, (tuple, list)):
            for net in model:
                net.eval()
        else:
            model.eval()

    test_iter = make_iterator(test=True, string='Evaluating (epoch {}): '.format(epoch))
    results = {}
    samples_ = None
    for inputs in test_iter:
        samples__ = {}
        for k, v in exp.PROCEDURES.items():
            if k == 'main' or not isinstance(exp.ARGS['procedures'], dict):
                args = exp.ARGS['procedures']
            else:
                args = exp.ARGS['procedures'][k]
            loss, results_, samples, condition = v(exp.MODELS, inputs, **args)
            if not samples_ and samples:
                samples__.update(**samples)
            update_dict_of_lists(results, **results_)
        samples_ = samples_ or samples__

    results = dict((k, np.mean(v)) for k, v in results.items())

    return results, samples_


def main_loop(summary_updates=None, epochs=None, updates_per_model=None, archive_every=None):
    info = pprint.pformat(exp.ARGS)
    viz.visualizer.text(info, env=exp.NAME, win='info')
    try:
        for e in xrange(epochs):
            epoch = exp.INFO['epoch']

            start_time = time.time()
            train_results_ = train_epoch(epoch)
            update_dict_of_lists(exp.SUMMARY['train'], **train_results_)

            test_results_, samples_ = test_epoch(epoch)
            update_dict_of_lists(exp.SUMMARY['test'], **test_results_)

            logger.info(' | '.join(['{}: {:.2f}/{:.2f}'.format(k, train_results_[k], test_results_[k])
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
                response = raw_input('Keyboard interrupt. Kill? (Y/N) '
                                     '(or ^c again)')
            except KeyboardInterrupt:
                kill = True
                break
            response = response.lower()
            if response == 'y':
                kill = True
                break
            elif response == 'n':
                print 'Cancelling interrupt. Starting epoch over.'
                break
            else:
                print 'Unknown response'

        if kill:
            print('Training interrupted')
            exp.save(prefix='interrupted')
            sys.exit(0)

    exp.save(prefix='final')