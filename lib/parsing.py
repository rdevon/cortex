'''Argument parsing module

'''

__author__ = 'R Devon Hjelm'
__author_email__ = 'erroneus@gmail.com'

import argparse
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
    parser.add_argument('arch', type=str, help='Architecture name')
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
    parser.add_argument('-a', '--args', default=None, type=str,
                        help=('Arguments for the main file'))
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
    parser.add_argument('-t', '--test', action='store_true', default=False)
    parser.add_argument('-d', '--device', type=int, default=0)
    return parser


def update_args(args, **kwargs):
    if args is not None:
        a = args.split(',')
        for a_ in a:
            k, v = a_.split('=')

            try:
                v = ast.literal_eval(v)
            except ValueError:
                pass

            k_split = k.split('.')
            kw = kwargs
            k_base = None
            for i, k_ in enumerate(k_split):
                if i < len(k_split) - 1:
                    if k_base in _known_args and k_ not in kw:
                        if k_ in _known_args[k_base]:
                            kw[k_] = {}
                    if k_ in kw:
                        kw = kw[k_]
                        k_base = k_
                    else:
                        raise ValueError('Unknown arg {}'.format(k))
                else:
                    if kw is None:
                        kw = dict(k_=v)
                    else:
                        kw[k_] = v

class StoreDictKeyPair(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        d = {}
        for kv in values.split(','):
            k, v = kv.split('=')
            d[k] = v
        setattr(namespace, self.dest, d)


def parse_args():
    # Parse args
    parser = make_argument_parser()
    for arg_k in _args:
        args = _args[arg_k]
        for k, v in args.items():
            arg_str = '--' + arg_k + '.' + k
            if isinstance(v, dict):
                parser.add_argument(arg_str, default=None, dest=arg_k + '.' + k, action=StoreDictKeyPair,
                                    metavar='key1=val1,key2=val2...')
            else:
                type_ = type(v) if v is not None else str
                parser.add_argument(arg_str, default=None, type=type_, help=_args_help[arg_k][k])
    args = parser.parse_args()

    return args