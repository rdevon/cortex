from cortex.built_ins.models.classifier import ImageClassification
from cortex._lib import (config, data, optimizer, setup_experiment, train)
from cortex._lib.utils import print_section
from torch.nn import CrossEntropyLoss
from argparse import Namespace
import torch


def test_argparsing():
    args = Namespace(
        classifier_args={'dropout': 0.2},
        classifier_type='convnet',
        clean=False,
        command=None,
        config_file=None,
        criterion=CrossEntropyLoss(),
        device=0,
        load_models=None,
        meta=None,
        name=None,
        out_path=None,
        reload=None,
        reloads=None,
        verbosity=1,
        **{
            'data.batch_size': 128,
            'data.copy_to_local': True,
            'data.data_args': None,
            'data.inputs': None,
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
            'train.train_mode': 'train'
        })
    args_data = {
        'source': 'CIFAR10',
        'batch_size': 128,
        'n_workers': 4,
        'skip_last_batch': False,
        'inputs': {
            'inputs': 'images'
        },
        'copy_to_local': True,
        'data_args': {},
        'shuffle': True
    }
    args_train = {
        'epochs': 1,
        'archive_every': 10,
        'quit_on_bad_values': True,
        'save_on_best': 'losses.classifier',
        'save_on_lowest': None,
        'save_on_highest': None,
        'eval_during_train': True,
        'train_mode': 'train',
        'test_mode': 'test',
        'eval_only': False
    }
    args_optimizer = {
        'optimizer': 'Adam',
        'learning_rate': 0.001,
        'weight_decay': {},
        'clipping': {},
        'optimizer_options': {},
        'model_optimizer_options': {}
    }

    model = ImageClassification()
    # args = setup_cortex(model=model)
    config.set_config()
    print_section('EXPERIMENT')
    model = setup_experiment(args, model=model)
    print_section('DATA')
    data.setup(args_data)
    print_section('NETWORKS')
    model.build()
    if args.load_models:
        d = torch.load(args.load_models, map_location='cpu')
        for k in args.reloads:
            model.nets[k].load_state_dict(d['nets'][k].state_dict())
    print_section('OPTIMIZER')
    optimizer.setup(model, args_optimizer)
    train.main_loop(model, args_train)
