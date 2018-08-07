from cortex.built_ins.models.classifier import ImageClassification
from cortex._lib import (config, data, exp, optimizer, setup_cortex,
                         setup_experiment, train)
from cortex._lib.utils import print_section
from argparse import Namespace
import torch.nn as nn

args = Namespace(
        classifier_args={'dropout': 0.2},
        classifier_type='convnet',
        clean=False,
        command=None,
        config_file=None,
        criterion=nn.CrossEntropyLoss(),
        device=0,
        load_models=None,
        meta=None,
        name=None,
        out_path=None,
        reload=None,
        reloads=None,
        verbosity=1,
        **{'data.batch_size': 128,
           'data.copy_to_local': True,
           'data.data_args': None,
           'data.inputs': {'inputs':'images'},
           'data.n_workers': 4,
           'data.shuffle': True,
           'data.skip_last_batch': False,
           'data.source': 'CIFAR10',
           'optimizer.clipping': None,
           'optimizer.learning_rate': 0.001,
           'optimizer.model_optimizer_options': None,
           'optimizer.optimizer': 'Adam',
           'optimizer.optimizer_options': None,
           'optimizer.weight_decay': None,
           'train.archive_every': 10,
           'train.epochs': 1,
           'train.eval_during_train': True,
           'train.eval_only': False,
           'train.quit_on_bad_values': True,
           'train.save_on_best': 'losses.classifier',
           'train.save_on_highest': None,
           'train.save_on_lowest': None,
           'train.test_mode': 'test',
           'train.train_mode': 'train'}
    )


classifier = ImageClassification()

# model_bce_loss = ClassifierBCELoss()
model = ImageClassification()
# args = setup_cortex(model=model)
config.set_config()
print_section('EXPERIMENT')
model = setup_experiment(args, model=model)
print_section('DATA')
data.setup(**exp.ARGS['data'])
print_section('NETWORKS')
model.build()
print_section('OPTIMIZER')
optimizer.setup(model, **exp.ARGS['optimizer'])
# train.main_loop(model, **exp.ARGS['train'])

nets = classifier.nets
# print(dir(classifier))
import pdb; pdb.set_trace()
print(nets['classifier'])
# print(nets.parameters)