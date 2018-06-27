'''Argument parsing module

'''

import argparse
import ast
import inspect
import logging
import re

from sphinxcontrib.napoleon import Config
from sphinxcontrib.napoleon.docstring import GoogleDocstring

from . import data, optimizer, train

__author__ = 'R Devon Hjelm'
__author_email__ = 'erroneus@gmail.com'


def parse_kwargs(f):
    kwargs = {}
    sig = inspect.signature(f)
    for i, (sk, sv) in enumerate(sig.parameters.items()):

        if sk == 'self':
            pass
        elif sk == 'kwargs':
            pass
        elif sv.default == inspect.Parameter.empty:
            pass
        else:
            kwargs[sv.name] = sv.default
    return kwargs


def parse_inputs(f):
    args = []
    sig = inspect.signature(f)
    for i, (sk, sv) in enumerate(sig.parameters.items()):

        if sk == 'self':
            pass
        elif sk == 'kwargs':
            pass
        elif sv.default == inspect.Parameter.empty:
            args.append(sv.name)
    return args


def parse_docstring(f):
    if f.__doc__ is None:
        return {}
    doc = inspect.cleandoc(f.__doc__)
    config = Config()
    google_doc = GoogleDocstring(doc, config)
    rst = str(google_doc)
    param_regex = r':param (?P<param>\w+): (?P<doc>.*)'
    m = re.findall(param_regex, rst)
    args_help = dict((k, v) for k, v in m)
    return args_help


def parse_header(f):
    doc = inspect.cleandoc(f.__doc__)
    config = Config()
    google_doc = GoogleDocstring(doc, config)
    rst = str(google_doc)
    lines = [l for l in rst.splitlines() if len(l) > 0]
    if len(lines) >= 2:
        return lines[:2]
    elif len(lines) == 1:
        return lines[0]
    else:
        return None, None


data_help = parse_docstring(data.setup)
data_args = parse_kwargs(data.setup)
train_help = parse_docstring(train.main_loop)
train_args = parse_kwargs(train.main_loop)
optimizer_help = parse_docstring(optimizer.setup)
optimizer_args = parse_kwargs(optimizer.setup)

default_args = dict(data=data_args, optimizer=optimizer_args, train=train_args)
default_help = dict(data=data_help, optimizer=optimizer_help, train=train_help)

_protected_args = ['arch', 'out_path', 'name', 'reload',
                   'args', 'copy_to_local', 'meta', 'config_file',
                   'clean', 'verbosity', 'test']

logger = logging.getLogger('cortex.parsing')


def make_argument_parser():
    '''Generic experiment parser.

    Generic parser takes the experiment yaml as the main argument, but has some
    options for reloading, etc. This parser can be easily extended using a
    wrapper method.

    Returns:
        argparse.parser

    '''
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-o',
        '--out_path',
        default=None,
        help=('Output path directory. All model results will go'
              ' here. If a new directory, a new one will be '
              'created, as long as parent exists.'))
    parser.add_argument(
        '-n',
        '--name',
        default=None,
        help=('Name of the experiment. If given, base name of '
              'output directory will be `--name`. If not given,'
              ' name will be the base name of the `--out_path`'))
    parser.add_argument('-r', '--reload', type=str, default=None,
                        help=('Path to model to reload.'))
    parser.add_argument('-R', '--reloads', type=str, nargs='+', default=None)
    parser.add_argument('-M', '--load_models',
                        type=str, default=None,
                        help=('Path to model to reload. Does not load args,'
                              ' info, etc'))
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


def parse_args(models):
    # Parse args
    parser = make_argument_parser()

    subparsers = parser.add_subparsers(
        title='Cortex',
        help='Select an architecture.',
        description='Cortex is a wrapper '
                    'around pytorch that makes training models '
        'more convenient.',
        dest='command')

    subparsers.add_parser(
        'setup', help='Setup cortex configuration.',
        description='Initializes or updates the `.cortex.yml` file.')

    for k, model in models.items():
        model_help, model_description = parse_header(model)

        subparser = subparsers.add_parser(
            k,
            help=model_help,
            description=model_description,
            formatter_class=lambda prog: argparse.HelpFormatter(
                prog,
                max_help_position=50,
                width=100))
        kwargs = model.kwargs
        helps = model.help

        for k, v in kwargs.items():
            arg_str = '--' + k
            help = helps.get(k, None)
            choices = None

            if isinstance(v, dict):
                subparser.add_argument(
                    arg_str,
                    dest=k,
                    default=v,
                    action=StoreDictKeyPair,
                    help=help,
                    metavar='<k1=v1,,k2=v2...>')
            elif isinstance(v, bool) and not v:
                action = 'store_true'
                subparser.add_argument(
                    arg_str, dest=k, action=action, default=False, help=help)
            elif isinstance(v, bool):
                type_ = type(v)
                metavar = '<' + type_.__name__ + '> (default=' + str(v) + ')'
                subparser.add_argument(
                    arg_str,
                    dest=k,
                    default=True,
                    metavar=metavar,
                    type=str2bool,
                    help=help)
            else:
                type_ = type(v) if v is not None else str
                metavar = '<' + type_.__name__ + '> (default=' + str(v) + ')'
                subparser.add_argument(
                    arg_str,
                    dest=k,
                    choices=choices,
                    metavar=metavar,
                    default=v,
                    type=type_,
                    help=help)

        for key, args in default_args.items():
            for k, v in args.items():
                arg_str = '--' + key[0] + '.' + k
                help = default_help[key][k]
                dest = key + '.' + k
                metavar = '<k1=v1,,k2=v2...>'
                if isinstance(v, dict):
                    subparser.add_argument(
                        arg_str,
                        dest=dest,
                        default=None,
                        action=StoreDictKeyPair,
                        help=help,
                        metavar=metavar)
                elif isinstance(v, bool) and not v:
                    action = 'store_true'
                    dest = key + '.' + k
                    subparser.add_argument(arg_str, dest=dest,
                                           action=action, default=False,
                                           help=help)
                elif isinstance(v, bool):
                    type_ = type(v)
                    metavar = '<' + type_.__name__ + \
                        '> (default=' + str(v) + ')'
                    dest = key + '.' + k
                    subparser.add_argument(
                        arg_str,
                        dest=dest,
                        default=True,
                        metavar=metavar,
                        type=str2bool,
                        help=help)
                else:
                    type_ = type(v) if v is not None else str
                    metavar = '<' + type_.__name__ + \
                        '> (default=' + str(v) + ')'
                    subparser.add_argument(
                        arg_str,
                        dest=dest,
                        default=None,
                        metavar=metavar,
                        type=type_,
                        help=help)

    args = parser.parse_args()

    return args
