'''Testing for building models.

'''

from cortex._lib.models import MODEL_PLUGINS
from cortex.built_ins.networks.fully_connected import FullyConnectedNet
from cortex.plugins import ModelPlugin, register_model


def test_class(model_class, arguments):
    '''Tests simple class attributions.

    Args:
        model_class: ModulePlugin subclass.
        arguments: Arguments for the class.

    '''
    arg1 = arguments['arg1']
    arg2 = arguments['arg2']
    arg1_help = arguments['arg1_help']
    arg2_help = arguments['arg2_help']

    assert model_class._help[arg1] == arg1_help, model_class.help[arg1]
    assert model_class._help[arg2] == arg2_help, model_class.help[arg2]
    assert model_class._kwargs[arg1] == 17, model_class.kwargs[arg1]
    assert model_class._kwargs[arg2] == 19, model_class.kwargs[arg2]


def test_register(model_class):
    '''Tests registration of a model.

    Args:
        model_class: ModelPlugin subclass.

    '''

    MODEL_PLUGINS.clear()
    register_model(model_class)
    assert isinstance(list(MODEL_PLUGINS.values())[0], model_class)
    MODEL_PLUGINS.clear()


def test_build(model_class, arguments):
    '''Tests building the model.

    Args:
        model_class: ModulePlugin subclass.
        arguments: Arguments for the class.

    '''
    ModelPlugin._all_nets.clear()
    kwargs = {arguments['arg1']: 11, arguments['arg2']: 13}

    model = model_class()
    model.kwargs.update(**kwargs)

    model.build()

    print('Model networks:', model.nets)
    assert isinstance(model.nets.net, FullyConnectedNet)

    parameters = list(model.nets.net.parameters())

    print('Parameter sizes:', [p.size() for p in parameters])
    assert parameters[0].size(1) == 11
    assert parameters[2].size(0) == parameters[3].size(0) == 13


def test_subplugin(model_class_with_submodel):
    '''Tests a model with a model inside.

    Args:
        model_class_with_submodel: ModulePlugin subclass.

    '''

    contract = dict(
        kwargs=dict(b='c'),
        nets=dict(net='net')
    )

    model = model_class_with_submodel(sub_contract=contract)

    try:
        model.build()
        assert 0
    except KeyError as e:
        print('build failed ({}). This is expected.'.format(e))

    ModelPlugin._all_nets.clear()

    sub_contract = dict(
        kwargs=dict(a='c'),
        nets=dict(net='net2')
    )
    model = model_class_with_submodel(sub_contract=sub_contract)

    model.build()
