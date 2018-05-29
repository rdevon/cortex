'''Main file for running experiments.

'''

__author__ = 'R Devon Hjelm'
__author_email__ = 'erroneus@gmail.com'

import logging

from cortex._lib import setup_cortex, exp
from cortex._lib.data import setup as setup_data, DATA_HANDLER
from cortex._lib.models import setup_model
from cortex._lib.optimizer import setup as setup_optimizer
from cortex._lib.train import setup as setup_train, main_loop
from cortex._lib.utils import print_section


logger = logging.getLogger('cortex')


def main():
    '''Main function.

    '''
    # Parse the command-line arguments

    try:
        print_section('LOADING DATA') ##############################################
        setup_data(**exp.ARGS.data)

        print_section('MODEL') #####################################################
        setup_model(DATA_HANDLER, **exp.ARGS.model)

        print_section('OPTIMIZER') #################################################
        setup_optimizer(**exp.ARGS.optimizer)

        print_section('TRAIN') #####################################################
        setup_train()

    except KeyboardInterrupt:
        print('Cancelled')
        exit(0)

    main_loop(**exp.ARGS.train)