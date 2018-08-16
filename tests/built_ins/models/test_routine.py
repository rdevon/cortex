'''Module for testing model routines.

'''

import torch.optim as optim

from cortex.plugins import ModelPlugin


def test_routine(model_class, arguments, data_class):
    ModelPlugin._reset_class()

    kwargs = {arguments['arg1']: 11, arguments['arg2']: 13}
    data = data_class(11)

    model = model_class(contract=dict(inputs=dict(A='test')))
    model._data = data
    model.kwargs.update(**kwargs)

    model.build()

    model.eval_step()
    print('Training nets: ', model._training_nets)
    assert 'net' in list(model._training_nets.values())[0]

    params = list(model.nets.net.parameters())
    op = optim.SGD(params, lr=0.0001)
    model._optimizers = dict(net=op)

    A = model.inputs('A')
    model.routine(A)

    model._reset_epoch()

    model.train_step()
    model.train_step()
    model.train_step()

    print('Results:', model._all_epoch_results)
    print('Losses:', model._all_epoch_losses)
    print('Times:', model._all_epoch_times)

    assert len(list(model._all_epoch_results.values())[0]) == 3
    assert len(list(model._all_epoch_losses.values())[0]) == 3
    assert len(list(model._all_epoch_times.values())[0]) == 3


def test_routine_with_submodels(model_with_submodel):
    model = model_with_submodel
    model.build()

    params = list(model.nets.net.parameters())
    op = optim.SGD(params, lr=0.0001)

    params2 = list(model.nets.net2.parameters())
    op2 = optim.SGD(params2, lr=0.001)

    model._optimizers = dict(net=op, net2=op2)
    model.submodel._optimizers = dict(net=op, net2=op2)

    assert model._get_training_nets() == []
    model.train_step()
    assert model._get_training_nets() == ['net', 'net2']
    model.train_step()
    model.train_step()


def test_routine_with_submodels_2(model_class_with_submodel_2, data_class):
    ModelPlugin._reset_class()

    kwargs = {'d': 11, 'c': 13}
    data = data_class(11)

    contract = dict(inputs=dict(B='test'))
    sub_contract = dict(
        kwargs=dict(a='d'),
        nets=dict(net='net2'),
        inputs=dict(A='test')
    )

    sub_contract2 = dict(
        kwargs=dict(a='d'),
        nets=dict(net='net3'),
        inputs=dict(A='test')
    )

    model = model_class_with_submodel_2(sub_contract1=sub_contract,
                                        sub_contract2=sub_contract2,
                                        contract=contract)
    model._data = data
    model.submodel1._data = data
    model.submodel2._data = data
    model.kwargs.update(**kwargs)

    model.build()

    params = list(model.nets.net.parameters())
    op = optim.SGD(params, lr=0.0001)

    params2 = list(model.nets.net2.parameters())
    op2 = optim.SGD(params2, lr=0.001)

    params3 = list(model.nets.net3.parameters())
    op3 = optim.SGD(params3, lr=0.001)

    model._optimizers = dict(net=op, net2=op2, net3=op3)
    model.submodel1._optimizers = dict(net=op, net2=op2, net3=op3)
    model.submodel2._optimizers = dict(net=op, net2=op2, net3=op3)

    model.train_step()
    model.train_step()
    model.train_step()
