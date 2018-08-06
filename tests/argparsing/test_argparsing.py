from cortex._lib import (config, setup_experiment, exp)
from cortex.built_ins.models.utils import update_encoder_args
from cortex.built_ins.models.classifier import ImageClassification
from .args_mock import args


class ClassifierModified(ImageClassification):
    defaults = dict(
        data=dict(batch_size=128, inputs=dict(inputs='images')),
        optimizer=dict(optimizer='Adam', learning_rate=1e-3),
        train=dict(epochs=200, save_on_best='losses.classifier'),
        model=dict(classifier_type='resnet'
                   ))

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


def test_command_override_static():
    expected_type = 'resnet'
    args.__dict__['classifier_type'] = expected_type
    classifier_defaults = ImageClassification()
    config.set_config()
    # NOTE: exp.ARGS is being populated inside setup_experiment() call
    classifier_defaults = setup_experiment(args, model=classifier_defaults, testmode=True)
    assert exp.ARGS['model']['classifier_type'] == expected_type


def test_static_override_parameters():
    expected_type = 'resnet'
    classifier_resnet = ClassifierModified()
    config.set_config()
    classifier_resnet = setup_experiment(args, model=classifier_resnet)
    assert exp.ARGS['model']['classifier_type'] == expected_type


def test_nested_arguments():
    # expected_classifier_args = {'dropout': 0.2, 'dim_h': 100}
    # model = ClassifierModified()
    # model.build(classifier_args=dict(dim_h=100))
    # # model.defaults['classifier_args'] = dict(dim_h=100)
    # config.set_config()
    # model = setup_experiment(args, model=model, testmode=True)
    # print(exp.ARGS)
    # assert model.defaults['classifier_args'] == {'dropout': 0.2, 'dim_h': 100}
    # assert model.defaults['classifier_args'] == {'dropout': 0.2, 'dim_h': 100}

