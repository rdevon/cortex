'''Argument parsing module

'''

import argparse
import copy
import inspect
import logging
import re
import sys

from sphinxcontrib.napoleon import Config
from sphinxcontrib.napoleon.docstring import GoogleDocstring

from . import data, exp, optimizer, train

__author__ = 'R Devon Hjelm'
__author_email__ = 'erroneus@gmail.com'


def parse_kwargs(f):
    """Parses kwargs from a function definition.

    Args:
        f: Function to parse.

    Returns:
        A dictionary of kwargs.

    """
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
            v = copy.deepcopy(sv.default)
            kwargs[sv.name] = v
    return kwargs


def parse_annotation(f):
    """Parses kwargs types from a function definition.

    Args:
        f: Function to parse.

    Returns:
        A dictionary of types.

    """
    annotations = {}
    sig = inspect.signature(f)
    for i, (sk, sv) in enumerate(sig.parameters.items()):
        annotation = sv.annotation
        if sk == 'self':
            pass
        elif sk == 'kwargs':
            pass
        elif sv.default == inspect.Parameter.empty:
            pass
        elif annotation != inspect.Parameter.empty:
            annotations[sv.name] = annotation

    return annotations


def parse_inputs(f):
    """Parses input variables from function definition.

    Args:
        f: Function to parse.

    Returns:
        List of variable names.

    """
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
    """Parses a docstring from a function defintion.

    Args:
        f: Function to parse from.

    Returns:
        Docstring.

    """
    if f.__doc__ is None:
        f.__doc__ = 'TODO\n TODO'
    doc = inspect.cleandoc(f.__doc__)
    config = Config()
    google_doc = GoogleDocstring(doc, config)
    rst = str(google_doc)
    param_regex = r':param (?P<param>\w+): (?P<doc>.*)'
    m = re.findall(param_regex, rst)
    args_help = dict((k, v) for k, v in m)
    return args_help


def parse_header(f):
    """Parses a header from a function definition.

    Args:
        f: Function to parse from.

    Returns:
        Header string.

    """
    if f.__doc__ is None:
        doc = 'TODO\n TODO'
    else:
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

DEFAULT_ARGS = dict(data=data_args, optimizer=optimizer_args, train=train_args)
DEFAULT_HELP = dict(data=data_help, optimizer=optimizer_help, train=train_help)

_protected_args = ['arch', 'out_path', 'name', 'reload',
                   'args', 'copy_to_local', 'meta', 'config_file',
                   'clean', 'verbosity', 'test']

logger = logging.getLogger('cortex.parsing')


def make_argument_parser() -> argparse.ArgumentParser:
    '''Generic experiment parser.

    Generic parser takes the experiment yaml as the main argument, but has some
    options for reloading, etc. This parser can be easily extended using a
    wrapper method.

    Returns:
        argparse.parser

    '''
    parser = argparse.ArgumentParser(
        formatter_class=lambda prog: argparse.HelpFormatter(
            prog, max_help_position=50, width=100))
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
    parser.add_argument('-a', '--autoreload', default=False,
                        action='store_true')
    parser.add_argument('-R', '--networks_to_reload', type=str, nargs='+',
                        default=None)
    parser.add_argument('-L', '--load_networks',
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
    parser.add_argument('-d', '--device', type=int, nargs='+', default=0)
    parser.add_argument('-V', '--noviz', default=False, action='store_true', help='No visualization.')
    parser.add_argument('-vis', '--visdom', default='vis', type=str, help='options: vis, tb')
    return parser


def _parse_argument(values):
    # Puts quotes on things not currently in the Namespace
    while True:
        try:
            eval(values)
            break
        except NameError as e:
            name = str(e).split(' ')[1][1:-1]
            p = '(?<!\'){}(?!\')'.format(name)
            values = re.sub(p, "'{}'".format(name), values)

    d = eval(values)
    return d


class StoreDictKeyPair(argparse.Action):
    """Parses key value pairs from command line.

    """
    def __call__(self, parser, namespace, values, option_string=None):
        if '__' in values:
            raise ValueError('Private or protected values not allowed.')

        values_ = values

        try:
            d = _parse_argument(values)
        except:
            d = str(values_)

        setattr(namespace, self.dest, d)


def str2bool(v):
    """Converts a string to boolean.

    E.g., yes -> True, no -> False.

    Args:
        v: String to convert.

    Returns:
        True or False

    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def _parse_data(plugin, subparser, name):
    '''

    Args:
        plugin:
        subparser:

    '''
    hyperparameters = parse_kwargs(plugin.handle)
    infos = parse_docstring(plugin.handle)
    annotations = parse_annotation(plugin.handle)
    for k, v in hyperparameters.items():
        h_info = infos.get(k, None)
        dest = 'data_args.' + name + '.' + k
        arg_str = '--' + name + '_args.' + k
        _add_hyperparameter_argument(k, v, h_info, subparser, dest=dest, arg_str=arg_str,
                                     annotation=annotations.get(k, None))


def _parse_model(model, subparser, yaml_hypers=None):
    """Parses model definitions and adds arguments as command-line arguments.

    Args:
        model: Model to parse.
        subparser: argparse subparses to parse values to.
        yaml_hypers: Hyperparameters from config file.

    """
    global default_args

    def _flatten_args(args, d, prefix=None):
        '''Flatten to dot notation.

        '''
        for k, v in args.items():
            if isinstance(v, dict):
                if prefix is None:
                    prefix_ = k
                else:
                    prefix_ = '{}.{}'.format(prefix, k)
                _flatten_args(v, d, prefix=prefix_)
            else:
                if prefix is None:
                    key = k
                else:
                    key = prefix + '.' + k
                d[key] = v

    def _flatten(args):
        new_args = {}
        _flatten_args(args, new_args)
        return new_args

    # First pull the model hyperparameters as a dictionary
    hyperparameters = model.pull_hyperparameters()

    # Get default hyperparameters
    model_defaults_model = model.defaults.pop('hyperparameters', {})
    update_args(model_defaults_model, hyperparameters)

    # From config yaml file.
    yaml_hypers_train = yaml_hypers.pop('train', {})
    yaml_hypers_data = yaml_hypers.pop('data', {})
    yaml_hypers_optimizer = yaml_hypers.pop('optimizer', {})

    update_args(yaml_hypers, hyperparameters)

    # Get docstring information.
    info = model.pull_info()

    # Put hyperparameters into dot notation.
    hyperparameters = _flatten(hyperparameters)
    info = _flatten(info)

    # Loop through hypers then add to argparse
    hyperparameter_keys = sorted(list(hyperparameters.keys()))
    for k in hyperparameter_keys:
        v = hyperparameters[k]
        h_info = info.get(k, None)
        _add_hyperparameter_argument(k, v, h_info, subparser)

    # Update total hyperparameter arguments.
    default_args = dict((k, v) for k, v in DEFAULT_ARGS.items())
    update_args(model.defaults, default_args)
    update_args(dict(train=yaml_hypers_train, data=yaml_hypers_data, optimizer=yaml_hypers_optimizer), default_args)

    for key, args in default_args.items():
        _add_default_arguments(key, args, subparser)


def _add_default_arguments(key, args, subparser):
    """Parses the default values of a model for the command line.

    Args:
        key: Key of argument.
        args: values of argument.
        subparser: subparser to parse values to so values show up on command line..

    Returns:

    """
    for k, v in args.items():
        arg_str = '--' + key[0] + '.' + k

        try:
            help = DEFAULT_HELP[key][k]
            dest = key + '.' + k

            _add_hyperparameter_argument(k, v, help, subparser, dest=dest, arg_str=arg_str)
        except KeyError:
            pass


def _add_hyperparameter_argument(k, v, help, subparser, dest=None, arg_str=None, annotation=None):
    '''Adds hyperparameter to parser.

    Args:
        k: Name of the hyperparameter
        v: Default value.
        help: Info for the hyperparameter
        subparser: Subparser to add argument to.
        annotation: Type of the hyperparameter.

    '''
    arg_str = arg_str or '--' + k
    dest = dest or k

    type_ = annotation or type(v)
    if isinstance(v, (dict, list, tuple)):
        dstr = str(v)
        dstr = dstr.replace(' ', '')
        metavar = '<' + type_.__name__ + '>'
        if not ('[' in dstr or ']' in dstr):
            metavar += ' (default=' + dstr + ')'  # argparse doesn't like square brackets

        subparser.add_argument(
            arg_str,
            dest=dest,
            default=v,
            action=StoreDictKeyPair,
            help=help, metavar=metavar)
    elif isinstance(v, bool) and not v:
        action = 'store_true'
        subparser.add_argument(
            arg_str, dest=dest, action=action, default=False, help=help)
    elif isinstance(v, bool):
        metavar = '<' + type_.__name__ + '> (default=' + str(v) + ')'
        subparser.add_argument(
            arg_str,
            dest=dest,
            default=True,
            metavar=metavar,
            type=str2bool,
            help=help)
    elif v is None:
        if type_.__name__ != 'NoneType':
            metavar = '<' + type_.__name__ + '>'
        else:
            metavar = '<UNK type>'
        subparser.add_argument(
            arg_str,
            dest=dest,
            default=None,
            action=StoreDictKeyPair,
            help=help,
            metavar=metavar)
    else:
        metavar = '<' + type_.__name__ + '> (default=' + str(v) + ')'
        subparser.add_argument(
            arg_str,
            dest=dest,
            metavar=metavar,
            default=v,
            type=type_,
            help=help)


def parse_args(models, model=None):
    '''Parse the command line arguments.

    Args:
        models: dictionary of models.

    Returns:
        dictionary of kwargs.

    '''

    parser = make_argument_parser()

    command = sys.argv[1:]
    if '-c' in sys.argv or '--config' in sys.argv:

        if '-c' in command:
            c_idx = command.index('-c')
        else:
            c_idx = command.index('--config')

        if (len(command) <= c_idx + 1) or command[c_idx + 1].startswith('-'):
            raise ValueError('No argument provided after `-c` or `--config`')
        yaml = command[c_idx + 1]
        yaml_hypers = exp.from_yaml(config_file=yaml)
        command = command[:c_idx] + command[c_idx + 2:]
    else:
        yaml_hypers = {}

    if model is None:
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
                    prog, max_help_position=50, width=100))

            _parse_model(model, subparser, yaml_hypers=yaml_hypers)

    else:
        _parse_model(model, parser, yaml_hypers=yaml_hypers)

    if '--d.sources' in command:
        d_idx = command.index('--d.sources')
        if len(command) <= d_idx + 1 or command[d_idx + 1].startswith('-'):
            raise ValueError('No argument provided after `--d.sources`')
        arg = command[d_idx + 1]
        # Puts quotes on things not currently in the Namespace
        arg = _parse_argument(arg)
        plugins = data.get_plugins(arg)
        for name, (source, plugin) in plugins.items():
            _parse_data(plugin, parser, name=name)

    idx = []
    for i, c in enumerate(command):
        if c.startswith('-') and not(c.startswith('--')):
            idx.append(i)

    header = []

    # argparse is picky about ordering
    for i in idx[::-1]:
        a = None

        if i + 1 < len(command):
            a = command[i + 1]

        if a is not None and (a.startswith('-') or a.startswith('--')):
            a = None

        if a is not None:
            a = command.pop(i + 1)
            c = command.pop(i)
            header += [c, a]

        else:
            c = command.pop(i)
            header.append(c)

    command = header + command

    args = parser.parse_args(command)
    if not hasattr(args, 'command'):
        args.command = None

    return args


def update_args(kwargs, kwargs_to_update):
    """Updates kwargs from another set of kwargs.

    Does dictionary traversal.

    Args:
        kwargs: dictionary to update with.
        kwargs_to_update: dictionary to update.

    """
    def _update_args(from_kwargs, to_kwargs):
        for k, v in from_kwargs.items():
            if isinstance(v, dict) and k not in to_kwargs:
                to_kwargs[k] = v
            elif isinstance(v, dict) and isinstance(to_kwargs[k], dict):
                _update_args(v, to_kwargs[k])
            else:
                to_kwargs[k] = v

    _update_args(kwargs, kwargs_to_update)
