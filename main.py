'''Main file for running experiments.

'''

__author__ = 'R Devon Hjelm'
__author_email__ = 'erroneus@gmail.com'

import sys
if sys.version_info < (3, 0):
    sys.stdout.write('Cortex requires Python 3.x, Python 2.x not supported\n')
    sys.exit(1)

import logging
from os import path

from lib import setup_cortex, exp
from lib.config import update_config
from lib.data import setup as setup_data, DATA_HANDLER
from lib.models import setup_model
from lib.optimizer import setup as setup_optimizer
from lib.parsing import parse_args
from lib.train import setup as setup_train, main_loop
from lib.utils import print_section


logger = logging.getLogger('cortex')


def main(eval_mode=False):
    '''Main function.

    '''

    # Setup file paths
    config_file_path = path.join(path.dirname(
        path.abspath(__file__)), 'config.yaml')
    if not path.isfile(config_file_path):
        config_file_path = None

    # User config
    update_config(config_file_path)

    # Parse the command-line arguments
    args = parse_args()
    setup_cortex(args)

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