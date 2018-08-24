'''Main file for running experiments.

'''

import logging

from cortex._lib import (config, data, exp, optimizer, setup_cortex,
                         setup_experiment, train)
from cortex._lib.utils import print_section
from cortex._lib.viz_server import VizServerSingleton
from cortex._lib.config import _yes_no
import visdom


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
            model, reload_nets = setup_experiment(args, model=model)
            print_section('DATA')
            data.setup(**exp.ARGS['data'])
            print_section('MODEL')
            model.reload_nets(reload_nets)
            model.build()
            print_section('OPTIMIZER')
            optimizer.setup(model, **exp.ARGS['optimizer'])

    except KeyboardInterrupt:
        print('Cancelled')
        exit(0)

    train.main_loop(model, **exp.ARGS['train'])
    server = config.CONFIG.viz.get('server', None)
    port = config.CONFIG.viz.get('port', 8097)
    visualizer = visdom.Visdom(server=server, port=port)
    if visualizer.check_connection():
        if _yes_no(
                "Experiment is finished. Do you want to close Visdom server? "
                "Warning: closing the server can make you lose data."):
            viz_singleton = VizServerSingleton()
            viz_singleton.viz_process.terminate()



