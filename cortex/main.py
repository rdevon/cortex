'''Main file for running experiments.

'''


import logging

from cortex._lib import (config, data, exp, optimizer, setup_cortex,
                         setup_experiment, train)
from cortex._lib.utils import print_section

import torch

__author__ = 'R Devon Hjelm'
__author_email__ = 'erroneus@gmail.com'


logger = logging.getLogger('cortex')


def run(model=None):
    '''Main function.

    '''
    # Parse the command-line arguments

    try:
        args = setup_cortex(model=model)

        if args.command == 'setup':
            # Performs setup only.
            config.setup()
            exit(0)
        else:
            config.set_config()
            print_section('EXPERIMENT')
            model = setup_experiment(args, model=model)

            print_section('DATA')
            data.setup(**exp.ARGS['data'])

            print_section('NETWORKS')
            if args.reload and not args.load_models:
                pass
            else:
                model.build()
                if args.load_models:
                    d = torch.load(args.load_models)
                    for k in args.reloads:
                        model.nets[k].load_state_dict(d['nets'][k].state_dict())

            print_section('OPTIMIZER')
            optimizer.setup(model, **exp.ARGS['optimizer'])

    except KeyboardInterrupt:
        print('Cancelled')
        exit(0)

    train.main_loop(model, **exp.ARGS['train'])
