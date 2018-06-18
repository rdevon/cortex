'''Main file for running experiments.

'''


import logging

from cortex._lib import (config, data, exp, models, optimizer, setup_cortex,
                         setup_experiment, train)
from cortex._lib.utils import print_section


__author__ = 'R Devon Hjelm'
__author_email__ = 'erroneus@gmail.com'


logger = logging.getLogger('cortex')


def main():
    '''Main function.

    '''
    # Parse the command-line arguments

    try:
        args = setup_cortex()

        if args.command == 'setup':
            # Performs setup only.
            config.setup()
            exit(0)
        else:
            config.set_config()

            print_section('EXPERIMENT')
            setup_experiment(args)

            print_section('DATA')
            data.setup(**exp.ARGS.data)

            print_section('NETWORKS')
            models.build_networks()

            print_section('OPTIMIZER')
            optimizer.setup(**exp.ARGS.optimizer)

    except KeyboardInterrupt:
        print('Cancelled')
        exit(0)

    train.main_loop(**exp.ARGS.train)
