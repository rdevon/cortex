'''Module for training.

'''

import logging
import sys
import time

from . import exp
from .utils import bold

__author__ = 'R Devon Hjelm'
__author_email__ = 'erroneus@gmail.com'

logger = logging.getLogger('cortex.train')


def main_loop(model, epochs=500, archive_every=10, save_on_best=None,
              save_on_lowest=None, save_on_highest=None,
              train_mode='train', test_mode='test', eval_only=False,
              pbar_off=False, no_ascii=False):
    '''

    Args:
        epochs: Number of epochs.
        archive_every: Number of epochs for writing checkpoints.
        train_mode: Training data mode.
        test_mode: Testing data mode.
        save_on_lowest: Saves when lowest of this result is found.
        save_on_highest: Saves when highest of this result is found.
        eval_only: Gives results over a test epoch.
        pbar_off: Turn off the progressbar.
        no_ascii: If True, the do not display results to command line in ascii characters or color.

    '''

    logger.info('Starting main loop.')

    if eval_only:
        train_results_ = test_epoch(model, None, data_mode=train_mode,
                                    use_pbar=not (pbar_off))
        test_results_ = test_epoch(model, None, data_mode=test_mode,
                                  use_pbar=not(pbar_off))
        convert_to_numpy(train_results_)
        convert_to_numpy(test_results_)

        train_results_total = summarize_results(train_results_)
        test_results_total = summarize_results(test_results_)

        display_results(train_results_total, test_results_total, None,
                        None, exp.INFO['epoch'], 0, 0, 0, no_ascii=no_ascii)

        exit(0)

    epoch = exp.INFO['epoch']
    total_time = 0.

    while epoch < epochs:
        try:
            epoch = exp.INFO['epoch']
            logger.info('Epoch {} / {}'.format(epoch, epochs))
            start_time = time.time()

            # Training / Evaluation
            model.train_loop(epoch, data_mode='train', use_pbar=not(pbar_off))
            model.eval_loop(epoch, data_mode='test', use_pbar=not(pbar_off))

            #if save_on_best or save_on_highest or save_on_lowest:
            #    best = exp.save_best(model, best, save_on_best, save_on_lowest)

            epoch_time = time.time() - start_time
            total_time += epoch_time

            # Display epoch information
            print()
            print()
            s = 'Epoch {} / {} took {:.2f}s. Total time: {:.2f}s' \
                .format(epoch + 1, epochs, epoch_time, total_time)
            s = bold(s)
            print(s)

            exp.RESULTS.display(no_ascii=no_ascii)

            # Finishing up
            exp.INFO['epoch'] += 1
            if (archive_every and epoch % archive_every == 0):
                exp.save(model, prefix=epoch)
            else:
                exp.save(model, prefix='last')

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
                        print()
                        print('Cancelling interrupt. Starting epoch over.')
                        return False
                    else:
                        print()
                        print('Unknown response')

            kill = stop_training_query()

            if kill:
                print()
                print('Training interrupted')
                exp.save(model, prefix='interrupted')
                sys.exit(0)

    logger.info('Successfully completed training')
    exp.save(model, prefix='final')
