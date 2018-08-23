from cortex.built_ins.models.classifier import ImageClassification
from cortex._lib import (data, exp)


def test_image_classification(args):
    """
    NOTE: Building with default arguments image_classification.build()
          fails.
    TODO: Assert that the model has the right layers and hyperparameters.
    Args:
        args (@pytest.fixture): Namespace

    Asserts: model.defaults are the one wanted and the
             image_classification has the correct layers.

    """
    image_classification = ImageClassification()
    data.setup(**exp.ARGS['data'])
    expected_defaults = dict(
        data=dict(batch_size=128, inputs=dict(inputs='images')),
        optimizer=dict(optimizer='Adam', learning_rate=1e-3),
        train=dict(epochs=200, save_on_best='losses.classifier'))

    image_classification.build()
    assert image_classification.defaults == expected_defaults
