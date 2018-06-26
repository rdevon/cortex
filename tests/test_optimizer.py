'''Tests the optimizer functionality.

'''

from cortex._lib import optimizer
from cortex.plugins import ModelPlugin


'''
def test_optimizer(cls2):
    ModelPlugin._all_nets.clear()

    kwargs = {'d': 11, 'c': 13}
    data = DummyData(11)

    contract = dict(inputs=dict(B='test'))
    sub_contract = dict(
        kwargs=dict(a='d'),
        nets=dict(net='net2'),
        inputs=dict(A='test')
    )

    model = cls2(sub_contract=sub_contract, contract=contract)

    optimizer.setup()

    model._data = data
    model.submodel._data = data
    model.kwargs.update(**kwargs)

    inputs = model.get_inputs(model.build)
    kwargs = model.get_kwargs(model.build)
    model.build(*inputs, **kwargs)
'''