'''Argument parsing module

'''

__author__ = 'R Devon Hjelm'
__author_email__ = 'erroneus@gmail.com'

import argparse
import ast
import logging

from . import data, optimizer, train


_protected_args = ['arch', 'out_path', 'name', 'reload', 'args', 'copy_to_local', 'meta', 'config_file',
                  'clean', 'verbosity', 'test']

logger = logging.getLogger('cortex.parsing')

_args = dict(data=data._args, optimizer=optimizer._args, train=train._args)
_args_help = dict(data=data._args_help, optimizer=optimizer._args_help, train=train._args_help)


def make_argument_parser():
    '''Generic experiment parser.

    Generic parser takes the experiment yaml as the main argument, but has some
    options for reloading, etc. This parser can be easily extended using a
    wrapper method.

    Returns:
        argparse.parser

    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--out_path', default=None,
                        help=('Output path directory. All model results will go'
                              ' here. If a new directory, a new one will be '
                              'created, as long as parent exists.'))
    parser.add_argument('-n', '--name', default=None,
                        help=('Name of the experiment. If given, base name of '
                              'output directory will be `--name`. If not given,'
                              ' name will be the base name of the `--out_path`')
                        )
    parser.add_argument('-r', '--reload', type=str, default=None,
                        help=('Path to model to reload.'))
    parser.add_argument('-R', '--reloads', type=str, nargs='+', default=None)
    parser.add_argument('-M', '--load_models', type=str, default=None,
                        help=('Path to model to reload. Does not load args, info, etc'))
    parser.add_argument('-C', '--copy_to_local', action='store_true', default=False)
    parser.add_argument('-m', '--meta', type=str, default=None)
    parser.add_argument('-c', '--config_file', default=None,
                        help=('Configuration yaml file. '
                              'See `exps/` for examples'))
    parser.add_argument('-k', '--clean', action='store_true', default=False,
                        help=('Cleans the output directory. '
                              'This cannot be undone!'))
    parser.add_argument('-v', '--verbosity', type=int, default=1,
                        help='Verbosity of the logging. (0, 1, 2)')
    parser.add_argument('-d', '--device', type=int, default=0)
    return parser


class StoreDictKeyPair(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        d = {}
        
        for kv in values.split(',,'):
            k, v = kv.split('=')
            d[k] = ast.literal_eval(v)
        setattr(namespace, self.dest, d)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args(archs):
    # Parse args
    parser = make_argument_parser()

    subparsers = parser.add_subparsers(title='Cortex', help='Select an architecture.',
                                       description='Cortex is a wrapper around pytorch that makes training models '
                                                   'more convenient.',
                                       dest='arch')
    for k, arch in archs.items():
        subparser = subparsers.add_parser(k, help=arch.doc, description=arch.doc,
                                          formatter_class=lambda prog: argparse.HelpFormatter(
                                              prog, max_help_position=50, width=100))
        for k, v in arch.kwargs.items():
            arg_str = '--' + k
            info = arch.info.get(k, None)
            if info is not None:
                help = info.get('help', None)
                choices = info.get('choices', None)
            else:
                help = None
                choices = None

            if isinstance(v, dict):
                subparser.add_argument(arg_str, dest=k, default=v, action=StoreDictKeyPair,
                                       help=help, metavar='<k1=v1,,k2=v2...>')
            elif isinstance(v, bool) and not v:
                action = 'store_true'
                subparser.add_argument(arg_str, dest=k, action=action, default=False, help=help)
            elif isinstance(v, bool):
                type_ = type(v)
                metavar = '<' + type_.__name__ + '> (default=' + str(v) + ')'
                subparser.add_argument(arg_str, dest=k, default=True, metavar=metavar, type=str2bool, help=help)
            else:
                type_ = type(v) if v is not None else str
                metavar = '<' + type_.__name__ + '> (default=' + str(v) + ')'
                subparser.add_argument(arg_str, dest=k, choices=choices, metavar=metavar, default=v, type=type_,
                                       help=help)

        for arg_k in _args:
            args = _args[arg_k]
            for k, v in args.items():
                arg_str = '--' + arg_k[0] + '.' + k
                help = _args_help[arg_k][k]
                dest = arg_k + '.' + k
                metavar = '<k1=v1,,k2=v2...>'
                if isinstance(v, dict):
                    subparser.add_argument(arg_str, dest=dest, default=None, action=StoreDictKeyPair,
                                           help=help, metavar=metavar)
                elif isinstance(v, bool) and not v:
                    action = 'store_true'
                    dest = arg_k + '.' + k
                    subparser.add_argument(arg_str, dest=dest, action=action, default=False, help=help)
                elif isinstance(v, bool):
                    type_ = type(v)
                    metavar = '<' + type_.__name__ + '> (default=' + str(v) + ')'
                    dest = arg_k + '.' + k
                    subparser.add_argument(arg_str, dest=dest, default=True, metavar=metavar, type=str2bool, help=help)
                else:
                    type_ = type(v) if v is not None else str
                    metavar = '<' + type_.__name__ + '> (default=' + str(v) + ')'
                    subparser.add_argument(arg_str, dest=dest, default=None, metavar=metavar, type=type_, help=help)

    args = parser.parse_args()

    return args