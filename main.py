'''Main file for running experiments.

'''

__author__ = 'R Devon Hjelm'
__author_email__ = 'erroneus@gmail.com'

import sys
if sys.version_info < (3, 0):
    sys.stdout.write('Cortex requires Python 3.x, Python 2.x not supported\n')
    sys.exit(1)

import logging

from cortex import setup_cortex, exp
from cortex.data import setup as setup_data, DATA_HANDLER
from cortex.models import setup_model
from cortex.optimizer import setup as setup_optimizer
from cortex.train import setup as setup_train, main_loop
from cortex.utils import print_section


logger = logging.getLogger('cortex')


def main(eval_mode=False):
    '''Main function.

    '''
    # Parse the command-line arguments
    setup_cortex()

    print_section('LOADING DATA') ##############################################
    setup_data(**exp.ARGS.data)

    print_section('MODEL') #####################################################
    setup_model(DATA_HANDLER, **exp.ARGS.model)

    print_section('OPTIMIZER') #################################################
    setup_optimizer(**exp.ARGS.optimizer)

    if eval_mode:
        return
    print_section('TRAIN') #####################################################
    setup_train()
    main_loop(**exp.ARGS.train)


def reload_model(arch, model_file):
    setup_reload(arch, model_file)
    main(eval_mode=True)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Cancelled')
        exit(0)