'''Main file for running experiments.

'''

__author__ = 'R Devon Hjelm'
__author_email__ = 'erroneus@gmail.com'

import sys
if sys.version_info < (3, 0):
    sys.stdout.write('Cortex requires Python 3.x, Python 2.x not supported\n')
    sys.exit(1)

import logging

from cortex import setup_cortex, experiment
from cortex import setup as setup_data, DATA_HANDLER
from cortex import setup_model
from cortex import setup as setup_optimizer
from cortex import setup as setup_train, main_loop
from cortex import print_section

LOGGER = logging.getLogger('cortex')


def main(eval_mode=False):
    '''Main function.

    '''
    # Parse the command-line arguments
    setup_cortex()

    print_section('LOADING DATA')
    setup_data(**experiment.ARGS.data)

    print_section('MODEL')
    setup_model(DATA_HANDLER, **experiment.ARGS.model)

    print_section('OPTIMIZER')
    setup_optimizer(**experiment.ARGS.optimizer)

    if eval_mode:
        return
    print_section('TRAIN')
    setup_train()
    main_loop(**experiment.ARGS.train)


def reload_model(self, arch, model_file):
    self.setup_reload(arch, model_file)
    self.main(eval_mode=True)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Cancelled')
        exit(0)
