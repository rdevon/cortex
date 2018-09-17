'''Module for logging

'''

import logging

__author__ = 'R Devon Hjelm'
__author_email__ = 'erroneus@gmail.com'

logging.basicConfig()
logger = logging.getLogger('cortex')
logger.propagate = False

file_formatter = logging.Formatter(
    '%(asctime)s:%(name)s[%(levelname)s]: %(message)s\n')
stream_formatter = logging.Formatter(
    '[%(levelname)s:%(name)s]: %(message)s' + ' ' * 40 + '\n')


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
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.terminator = ''
    ch.setLevel(level)
    ch.setFormatter(stream_formatter)
    logger.addHandler(ch)
    logger.info('Setting logging to %s' % lstr)


def set_file_logger(file_path):
    global logger
    fh = logging.FileHandler(file_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(file_formatter)
    logger.addHandler(fh)
    fh.terminator = ''
