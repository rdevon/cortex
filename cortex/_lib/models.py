'''Builds arch

'''

__author__ = 'R Devon Hjelm'
__author_email__ = 'erroneus@gmail.com'

import importlib
import logging
import sys
import os

from .parsing import parse_docstring, parse_header, parse_kwargs
from .handlers import Handler, NetworkHandler


logger = logging.getLogger('cortex.models')
MODEL = None

_ROUTINE_PLUGINS = {}
_BUILD_PLUGINS = {}
MODEL_PLUGINS = {}
NETWORK_HANDLER = NetworkHandler()


def check_plugin(plugin, plugin_type_str, D):
    if plugin.plugin_name is None:
        ValueError('Set `plugin_name` static member for plugin.')
    if plugin.plugin_name in D:
        raise KeyError('plugin_name `{}` already registered as a {} plugin in cortex. '
                       'Try using another one.'.format(plugin_type_str, plugin.plugin_name))

    for k in plugin._protected:
        if hasattr(plugin, k):
            raise AttributeError('{} is a protected attribute.'.format(k))

    for k in plugin._required:
        v = getattr(plugin, k, None)
        if v is None:
            raise AttributeError('Plugin must have {} attribute set.'.format(k))
        else:
            setattr(plugin, k, v)


def register_build(plugin):
    global _BUILD_PLUGINS
    check_plugin(plugin, 'build', _BUILD_PLUGINS)

    plugin.help = parse_docstring(plugin.build)
    plugin.kwargs = parse_kwargs(plugin.build)

    _BUILD_PLUGINS[plugin.plugin_name] = plugin


def register_routine(plugin):
    global _ROUTINE_PLUGINS
    check_plugin(plugin, 'routine', _ROUTINE_PLUGINS)

    plugin.help = parse_docstring(plugin.run)
    plugin.kwargs = parse_kwargs(plugin.run)

    _ROUTINE_PLUGINS[plugin.plugin_name] = plugin


def register_model(plugin):
    global MODEL_PLUGINS
    check_plugin(plugin, 'model', MODEL_PLUGINS)

    plugin.help, plugin.description = parse_header(plugin)

    MODEL_PLUGINS[plugin.plugin_name] = plugin


class BuildReference():
    def __init__(self, key, **kwargs):
        self.reference = key
        self.kwargs = kwargs

    def resolve(self):
        try:
            plugin = _BUILD_PLUGINS[self.reference]
        except KeyError:
            raise KeyError('Build `{}` not registered in cortex. '
                           'Available: {}'.format(self.reference, _BUILD_PLUGINS.keys()))
        return plugin(**self.kwargs)

class RoutineReference():
    def __init__(self, key, **kwargs):
        self.reference = key
        self.kwargs = kwargs

    def resolve(self):
        try:
            plugin = _ROUTINE_PLUGINS[self.reference]
        except KeyError:
            raise KeyError('Routine `{}` not registered in cortex. '
                           'Available: {}'.format(self.reference, _ROUTINE_PLUGINS.keys()))
        return plugin(**self.kwargs)


class BuildPluginBase():
    pass


class RoutinePluginBase():
    _training_models = []
    pass


class ModelPluginBase():

    def check(self):
        for key in self.routines.keys():
            routine = self.routines[key]
            if isinstance(routine, RoutinePluginBase):
                pass
            elif isinstance(routine, RoutineReference):
                self.routines[key] = routine.resolve()
            else:
                raise ValueError

        for key in self.builds.keys():
            build = self.builds[key]
            if isinstance(build, BuildPluginBase):
                pass
            elif isinstance(build, BuildReference):
                self.builds[key] = build.resolve()
            else:
                raise ValueError

    def get_kwargs(self):
        kwargs = {}
        def add_kwargs(obj):
            for k, v in obj.kwargs.items():
                if k in kwargs:
                    if v is None:
                        pass
                    elif kwargs[k] is None:
                        kwargs[k] = v
                    elif kwargs[k] != v:
                        logger.warning('Multiple default values found for {}. This may have unintended '
                                       'effects. Using {}'.format(k, kwargs[k]))
                else:
                    kwargs[k] = v

        for build in self.builds.values():
            add_kwargs(build)

        for routine in self.routines.values():
            add_kwargs(routine)

        return kwargs

    def get_help(self):
        helps = {}
        def add_help(obj):
            for k, v in obj.help.items():
                if k in helps:
                    if v is None:
                        pass
                    elif helps[k] is None:
                        helps[k] = v
                    elif helps[k] != v:
                        logger.warning('Multiple default values found for {} help. This may have unintended '
                                       'effects. Using {}'.format(k, helps[k]))
                else:
                    helps[k] = v

        for build in self.builds.values():
            add_help(build)

        for routine in self.routines.values():
            add_help(routine)

        return helps

    def unpack_args(self):
        builds = Handler()
        routines = Handler()

        kwargs = self.get_kwargs()

        for key, build in self.builds.items():
            for k_, v in kwargs.items():
                if k_ in build.kwargs:
                    if key in builds:
                        builds[key][k_] = v
                    else:
                        builds[key] = {k_: v}

        for key, routine in self.routines.items():
            for k_, v in kwargs.items():
                if k_ in routine.kwargs:
                    if key in builds:
                        routines[key][k_] = v
                    else:
                        routines[key] = {k_: v}
        return Handler(builds=builds, routines=routines)


def setup_model(model_key):
    global MODEL
    logger.info('Using model `{}`'.format(model_key))
    MODEL = MODEL_PLUGINS[model_key]
    return MODEL


_arch_keys_optional = dict(
    TEST_ROUTINES='test_routines',
    FINISH_TRAIN_ROUTINES='finish_train_routines',
    FINISH_TEST_ROUTINES='finish_test_routines',
    SETUP='setup',
    Dataset='Dataset',
    DataLoader='DataLoader',
    transform='transform'
)

_ignore = ['__init__.py', '__pycache__']


def import_directory(p, name):
    '''
    Adds custom directories to the framwework
    '''

    global ARCHS

    if p.endswith('/'):
        p = p[:-1]

    logger.info('Adding {} to `sys.path`.'.format(p))
    sys.path.append(p)

    for fn in os.listdir(p):
        if fn.endswith('.py') and not fn in _ignore:
            fnp = fn[:-3]
            importlib.import_module(fnp)
            try:
                importlib.import_module(fnp)
            except Exception as e:
                logger.warning('Import of architecture (module) {} failed ({})'.format(fnp, e))

        '''
        elif os.path.isdir(fn):
            if fn.endswith('/'):
                fn = fn[:-1]
            if fn not in _ignore:
                add_directory(fn, name + '.' + os.path.basename(fn))
        '''

def find_models(model_paths):
    for k, p in model_paths.items():
        import_directory(p, k)

    global MODEL_PLUGINS
    keys = list(MODEL_PLUGINS.keys())
    for k in keys:
        v = MODEL_PLUGINS[k]
        v.check()
        try:
            v.check()
        except Exception as e:
            logger.warning('`{}` checks failed ({}).'.format(k, e))
            MODEL_PLUGINS.pop(k)

def build_networks(**build_args):
    '''Builds the generator and discriminator.

    If architecture module contains a `build_model` function, use that,
    otherwise, use the one found in this module.

    '''
    for build_key, build in MODEL.builds.items():
        args = build_args[build_key]
        logger.debug('{} build args: {}'.format(build_key, args))
        build(**args)

    MODEL.setup_routine_nets()


def reload_models(**reload_models):
    global MODEL_HANDLER
    if MODEL_HANDLER is None:
        raise RuntimeError('MODEL_HANDLER not set. `reload_models` should only be used after '
                           '`models.setup_models` has been called.')
    for k, v in reload_models.items():
        logger.info('Reloading model {}'.format(k))
        logger.debug(v)
        MODEL_HANDLER[k] = v
