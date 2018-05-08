'''Main file for running experiments.

'''

__author__ = 'R Devon Hjelm'
__author_email__ = 'erroneus@gmail.com'

import sys
if sys.version_info < (3, 0):
    sys.stdout.write('Cortex requires Python 3.x, Python 2.x not supported\n')
    sys.exit(1)

import logging

from __init__ import setup, setup_reload
from lib import exp
from lib.data import setup as setup_data, DATA_HANDLER
from lib.models import setup_model
from lib.train import setup as setup_optimizer, main_loop
from lib.utils import print_section


logger = logging.getLogger('cortex')


def main(eval_mode=False):
    '''Main function for continuous BGAN.

    '''
    data_args = exp.ARGS['data']
    model_args = exp.ARGS['model']
    optimizer_args = exp.ARGS['optimizer']
    train_args = exp.ARGS['train']

    print_section('LOADING DATA') ##############################################
    setup_data(**data_args)

    print_section('MODEL') #####################################################
    logger.info('Building model...')
    logger.info('Model args: {}'.format(model_args))
    models, routines = setup_model(DATA_HANDLER, **model_args)

    print_section('EXPERIMENT')
    exp.setup(models, **routines)

    print_section('OPTIMIZER') #################################################
    setup_optimizer(**optimizer_args)

    if eval_mode:
        return
    print_section('TRAIN') #####################################################
    main_loop(**train_args)


def reload_model(arch, model_file):
    import torch
    use_cuda = torch.cuda.is_available()
    setup_reload(arch, use_cuda, model_file)
    main(eval_mode=True)


if __name__ == '__main__':
    import torch
    use_cuda = torch.cuda.is_available()
    setup(use_cuda)

    try:
        main()
    except KeyboardInterrupt:
        print('Cancelled')
        exit(0)