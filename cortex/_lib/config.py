'''Module for config

'''

import glob
import logging
from os import path
import pathlib
import pprint
import readline
import socket
import yaml

from cortex._lib.handlers import Handler

logger = logging.getLogger('cortex.config')


class ConfigHandler(Handler):
    def __init__(self):
        super().__init__(viz={}, data_paths={}, arch_paths={}, out_path=None)


CONFIG = ConfigHandler()

_config_name = '.cortex.yml'
_welcome_message = 'Welcome to cortex! Cortex is a library meant to inject ' \
                   'your PyTorch code into the training loop, automating ' \
                   'common tasks such as optimization, visualization, and ' \
                   'checkpoint management.'
_info_message = 'Cortex requires a configuration file to run properly. \n' \
                'This will be stored in your home directory ({}).' \
                'If this is cancelled (using ^C), config file generation ' \
                'will be cancelled.'
_local_path_message = 'For many large datasets, you may want to copy ' \
                      'datasets to a local directory. If so, enter the path ' \
                      'here: [{}] '
_tv_path_message = 'Some built-in datasets rely on torchvision. ' \
                   'If you would like this functionality, enter the path ' \
                   'here (this can be an empty folder, as torchvision will ' \
                   'download the appropriate dataset): [{}] '
_data_message = 'Cortex can manage any of your datasets, so they can be ' \
                'simply referenced by name. If you have any datasets to add, ' \
                'you can enter them here (the reference name will be managed ' \
                'next). Otherwise, add them manually to ' \
                'the config file or run `cortex setup`.'
_data_path_message = 'Path to dataset (directory or file): (press Enter to ' \
                     'skip) '
_data_name_message = 'Name of dataset: [{}] '
_visdom_message = 'Cortex uses Visdom to do visualization. ' \
                  'This requires running a Visdom server separately ' \
                  '(see https://github.com/facebookresearch/visdom for ' \
                  'details).\n' \
                  'Note that running cortex is still possible without ' \
                  'visualization.'
_viz_ip_message = 'IP for the Visdom server: [{}] '
_viz_port_message = 'Visdom port: [{}] '
_out_message = 'Cortex requires an out path to store experiment files. ' \
               'This will be the location of binaries, as well as other ' \
               'results from experiments.'
_out_path_message = 'Enter the path to the output directory: [{}] '


def set_config():
    ''' Setups up cortex config.

    Reads from a configuration file. If file doesn't exist, this starts the
    config file setup.

    '''
    global CONFIG
    pathName = path.expanduser('~')
    config_file = path.join(pathName, _config_name)
    isfile = path.isfile(config_file)

    if isfile:
        logger.debug('Open config file {}'.format(config_file))
        with open(config_file, 'r') as f:
            d = yaml.load(f)
            logger.debug('User-defined configs: {}'.format(pprint.pformat(d)))

            viz = d.get('viz', {})
            data_paths = d.get('data_paths', {})
            arch_paths = d.get('arch_paths', {})
            out_path = d.get('out_path', None)

            CONFIG.update(viz=viz, data_paths=data_paths,
                          arch_paths=arch_paths, out_path=out_path)
    else:
        logger.warning('{} not found'.format(_config_name))
        setup_config_file(config_file)
        set_config()


def setup():
    pathName = path.expanduser('~')
    config_file = path.join(pathName, _config_name)
    setup_config_file(config_file)


def _complete_path(text, state):
    '''Completes a path for readline.

    '''
    return (glob.glob(text + '*') + [None])[state]


def _check_dir(query, default, required=False):
    while True:
        p = (input(query.format(default)) or default)

        if p is None:
            if required:
                print('Required path must be specified')
            else:
                return
        else:
            isdir = path.isdir(p)
            if isdir:
                return p
            else:
                create_path = _yes_no('Path not found at {}. Create?'
                                      .format(p))
                if create_path:
                    pathlib.Path(p).mkdir(parents=True, exist_ok=True)
                    return p


def _yes_no(query, default='no'):
    yes = ['yes', 'y', 'Yes', 'Y', 'YES']
    no = ['no', 'n', 'No', 'N', 'NO']
    query_ = query + ' (yes/no) [{}] '.format(default)
    while True:
        response = input(query_) or default
        if response in yes:
            return True
        elif response in no:
            return False
        else:
            print('Please enter `yes` or `no`')


def _query_dataset(d):
    data_path = input(_data_path_message) or None
    if data_path is None:
        return True

    is_file = path.isfile(data_path)
    is_dir = path.isdir(data_path)
    if is_file:
        default_name = path.basename(data_path).split('.')[-1]
    elif is_dir:
        if data_path.endswith('/'):
            data_path = data_path[:-1]
        default_name = path.basename(data_path)
    else:
        print('Data not found at {}'.format(data_path))
        return False

    while True:
        data_name = (input(_data_name_message.format(default_name)) or
                     default_name)
        if data_name in d['data_paths']:
            replace = _yes_no('{} already taken. Replace?'.format(data_name),
                              default='no')
            if replace:
                d['data_paths'][data_name] = data_path
                return True
        else:
            d['data_paths'][data_name] = data_path
            return True


def setup_config_file(config_file):
    isfile = path.isfile(config_file)

    if isfile:
        with open(config_file, 'r') as f:
            d = yaml.load(f)
    else:
        d = dict(data_paths={}, viz={}, out_path=None)

    readline.set_completer_delims(' \t\n;')
    readline.parse_and_bind('tab: complete')
    readline.set_completer(_complete_path)

    # Welcome
    print()
    print(_welcome_message)
    print()
    print(_info_message.format(config_file))
    print()

    # Local path
    local_default = d['data_paths'].get('local')
    local_path = _check_dir(_local_path_message, local_default)
    if local_path is not None:
        d['data_paths']['local'] = local_path
    print()

    # Torchvision path
    tv_default = d['data_paths'].get('torchvision')
    tv_path = _check_dir(_tv_path_message, tv_default)
    if tv_path is not None:
        d['data_paths']['torchvision'] = local_path
    print()

    # Extra data paths
    print(_data_message)
    while True:
        break_loop = _query_dataset(d)
        if break_loop:
            break
    print()
    print(_visdom_message)

    default_host = 'http://' + str(socket.gethostbyname(socket.gethostname()))
    default_host = d['viz'].get('server', default_host)
    default_port = 8097
    default_port = d['viz'].get('port', default_port)
    viz_ip = input(_viz_ip_message.format(default_host)) or default_host
    viz_port = input(_viz_port_message.format(default_port)) or default_port
    d['viz']['server'] = viz_ip
    d['viz']['port'] = viz_port
    print()

    print(_out_message)
    output_default = d.get('out_path')
    out_path = _check_dir(_out_path_message, output_default, required=True)
    d['out_path'] = out_path

    with open(config_file, 'w') as f:
        yaml.dump(d, f)
