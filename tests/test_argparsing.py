from cortex._lib import (config, setup_experiment, exp)
from cortex.built_ins.models.utils import update_encoder_args
from cortex.built_ins.models.classifier import ImageClassification
from tests.args_mock import args
from cortex._lib.parsing import update_args


class ClassifierModified(ImageClassification):
    defaults = dict(
        data=dict(batch_size=128, inputs=dict(inputs='images')),
        optimizer=dict(optimizer='Adam', learning_rate=1e-3),
        train=dict(epochs=200, save_on_best='losses.classifier'),
        model=dict(classifier_type='resnet'))

    def build(self,
              classifier_type='convnet',
              classifier_args=dict(dropout=0.2)):
        classifier_args = classifier_args or {}
        shape = self.get_dims('x', 'y', 'c')
        dim_l = self.get_dims('labels')
        Encoder, args = update_encoder_args(
            shape, model_type=classifier_type, encoder_args=classifier_args)
        args.update(**classifier_args)
        classifier = Encoder(shape, dim_out=dim_l, **args)
        self.nets.classifier = classifier


def update_nested_dicts(from_d, to_d):
    for k, v in from_d.items():
        if (k in to_d) and isinstance(to_d[k], dict):
            if not isinstance(v, dict):
                raise ValueError('Updating dict entry with non-dict.')
            update_nested_dicts(v, to_d[k])
        else:
            to_d[k] = v


def test_command_override_static():
    expected_type = 'resnet'
    args.__dict__['classifier_type'] = expected_type
    classifier_defaults = ImageClassification()
    config.set_config()
    # NOTE: exp.ARGS is being populated inside setup_experiment() call
    classifier_defaults = setup_experiment(
        args, model=classifier_defaults, testmode=True)
    assert exp.ARGS['model']['classifier_type'] == expected_type


def test_static_override_parameters():
    expected_type = 'resnet'
    classifier_resnet = ClassifierModified()
    config.set_config()
    classifier_resnet = setup_experiment(
        args, model=classifier_resnet, testmode=True)
    assert exp.ARGS['model']['classifier_type'] == expected_type


def test_update_nested_dicts():
    expected_classifier_args_before_update = {'dropout': 0.2}
    expected_classifier_args_after_update = {'dropout': 0.2, 'dim_h': 100}
    args_for_update = {
        'data': {
            'batch_size': 122
        },
        'model': {
            'classifier_args': {
                'dropout': 0.2,
                'dim_h': 100
            }
        }
    }
    model = ClassifierModified()
    config.set_config()
    model = setup_experiment(args, model=model, testmode=True)
    assert exp.ARGS['model'][
        'classifier_args'] == expected_classifier_args_before_update
    update_nested_dicts(args_for_update, exp.ARGS)
    assert exp.ARGS['model'][
        'classifier_args'] == expected_classifier_args_after_update
    assert exp.ARGS['data']['batch_size'] == 122


def test_update_args():
    print(exp.ARGS)
    expected_classifier_args_before_update = {'dropout': 0.2, 'dim_h': 100}
    expected_classifier_args_after_update = {'dropout': 0.1, 'dim_h': 100}
    args_for_update = {
        'data': {
            'batch_size': 128
        },
        'model': {
            'classifier_args': {
                'dropout': 0.1
            }
        }
    }
    model = ClassifierModified()
    config.set_config()
    model = setup_experiment(args, model=model, testmode=True)
    assert exp.ARGS['model'][
        'classifier_args'] == expected_classifier_args_before_update
    update_args(args_for_update, exp.ARGS)
    assert exp.ARGS['model'][
        'classifier_args'] == expected_classifier_args_after_update
    assert exp.ARGS['data']['batch_size'] == 128
