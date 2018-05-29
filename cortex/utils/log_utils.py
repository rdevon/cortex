'''Module for logging

'''

__author__ = 'R Devon Hjelm'
__author_email__ = 'erroneus@gmail.com'

import logging


logging.basicConfig()
LOGGER = logging.getLogger('cortex')
LOGGER.propagate = False

file_formatter = logging.Formatter(
    '%(asctime)s:%(name)s[%(levelname)s]: %(message)s\n')
stream_formatter = logging.Formatter(
    '[%(levelname)s:%(name)s]:%(message)s' + ' ' * 40 + '\n')


def set_stream_logger(verbosity):
    global logger

    if verbosity == 0:
        level = logging.WARNING
        lstr = 'WARNING'
    elif verbosity == 1:
        level = logging.INFO
        lstr = 'INFO'
    elif verbosity == 2:
        level = logging.DEBUG
        lstr = 'DEBUG'
    else:
        level = logging.INFO
        lstr = 'INFO'
    LOGGER.setLevel(level)
    ch = logging.StreamHandler()
    ch.terminator = ''
    ch.setLevel(level)
    ch.setFormatter(stream_formatter)
    LOGGER.addHandler(ch)
    LOGGER.info('Setting logging to %s' % lstr)


def set_file_logger(file_path):
    global logger
    fh = logging.FileHandler(file_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(file_formatter)
    LOGGER.addHandler(fh)
    fh.terminator = ''
    LOGGER.info('Saving logs to %s' % file_path)