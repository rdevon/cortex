'''Testing for building models.

'''

from cortex._lib.models import MODEL_PLUGINS
from cortex.built_ins.networks.fully_connected import FullyConnectedNet
from cortex.plugins import ModelPlugin, register_model


def test_class(model_class, arguments):
    arg1 = arguments['arg1']
    arg2 = arguments['arg2']
    arg1_help = arguments['arg1_help']
    arg2_help = arguments['arg2_help']

    assert model_class._help[arg1] == arg1_help, model_class.help[arg1]
    assert model_class._help[arg2] == arg2_help, model_class.help[arg2]
    assert model_class._kwargs[arg1] == 17, model_class.kwargs[arg1]
    assert model_class._kwargs[arg2] == 19, model_class.kwargs[arg2]


def test_register(model_class):
    register_model(model_class)
    assert model_class in MODEL_PLUGINS.values()
    MODEL_PLUGINS.clear()


def test_build(model_class, arguments):
    ModelPlugin._all_nets.clear()
    kwargs = {arguments['arg1']: 11, arguments['arg2']: 13}

    model = model_class()
    model.kwargs.update(**kwargs)

    model.build()

    print(model.nets)
    assert isinstance(model.nets.net, FullyConnectedNet)


def test_subplugin(model_class_with_submodel):

    contract = dict(
        kwargs=dict(b='c'),
        nets=dict(net='net')
    )

    model = model_class_with_submodel(sub_contract=contract)

    kwargs = model.get_kwargs(model.build)
    try:
        model.build(**kwargs)
        assert 0
    except KeyError:
        pass

    ModelPlugin._all_nets.clear()

    sub_contract = dict(
        kwargs=dict(a='c'),
        nets=dict(net='net2')
    )
    model = model_class_with_submodel(sub_contract=sub_contract)

    kwargs = model.get_kwargs(model.build)
    model.build(**kwargs)
    print(model.nets['net'].parameters())