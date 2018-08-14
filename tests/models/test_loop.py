'''Tests the training loop.

'''

from cortex._lib import optimizer


def test_loop(model_with_submodel):
    model = model_with_submodel
    model.build()
    optimizer.setup(model)

    model.train_loop(0)

    results = model._all_epoch_results

    print(results)

    rlen = len(results['TestModel2_output'])

    assert len(results['TestModel2_output']) == \
        len(results.losses['net']) == \
        len(results.times['TestModel2'])

    model.train_loop(0)

    results = model._all_epoch_results

    print(results)

    assert len(results['TestModel2_output']) == \
        len(results.losses['net']) == \
        len(results.times['TestModel2'])

    assert rlen == len(results['TestModel2_output'])
