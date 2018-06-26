'''Module for testing model routines.

'''

import torch.optim as optim

from cortex.plugins import ModelPlugin


def test_routine(model_class, arguments, data_class):
    ModelPlugin._all_nets.clear()

    kwargs = {arguments['arg1']: 11, arguments['arg2']: 13}
    data = data_class(11)

    contract = dict(inputs=dict(A='test'))
    model = model_class(contract=contract)
    model._data = data
    model.kwargs.update(**kwargs)

    inputs = model.get_inputs(model.build)
    kwargs = model.get_kwargs(model.build)
    model.build(*inputs, **kwargs)

    inputs = model.get_inputs(model.routine)
    kwargs = model.get_kwargs(model.routine)

    try:
        model.routine(*inputs, **kwargs)
        assert 0  # optimizer hasn't been set
    except KeyError:
        pass

    assert 'net' in list(model._training_nets.values())[0], model._training_nets

    params = list(model.nets.net.parameters())
    op = optim.SGD(params, lr=0.0001)
    model._optimizers = dict(net=op)

    model.routine(*inputs, **kwargs)

    model._reset_epoch()

    model.train_step()
    model.train_step()
    model.train_step()

    print(model._all_epoch_results)
    print(model._all_epoch_losses)
    print(model._all_epoch_times)

    assert len(list(model._all_epoch_results.values())[0]) == 3
    assert len(list(model._all_epoch_losses.values())[0]) == 3
    assert len(list(model._all_epoch_times.values())[0]) == 3


def test_routine_with_submodels(model_class_with_submodel, data_class):

    ModelPlugin._all_nets.clear()

    kwargs = {'d': 11, 'c': 13}
    data = data_class(11)

    contract = dict(inputs=dict(B='test'))
    sub_contract = dict(
        kwargs=dict(a='d'),
        nets=dict(net='net2'),
        inputs=dict(A='test')
    )

    model = model_class_with_submodel(sub_contract=sub_contract,
                                      contract=contract)
    model._data = data
    model.submodel._data = data
    model.kwargs.update(**kwargs)

    inputs = model.get_inputs(model.build)
    kwargs = model.get_kwargs(model.build)
    model.build(*inputs, **kwargs)

    params = list(model.nets.net.parameters())
    op = optim.SGD(params, lr=0.0001)

    params2 = list(model.nets.net2.parameters())
    op2 = optim.SGD(params2, lr=0.001)

    model._optimizers = dict(net=op, net2=op2)
    model.submodel._optimizers = dict(net=op, net2=op2)

    model.train_step()
    model.train_step()
    model.train_step()