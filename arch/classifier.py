'''Simple classifier model

'''


import torch
import torch.nn as nn
import torch.nn.functional as F


resnet_args_ = dict(dim_h=64, batch_norm=True, f_size=3, n_steps=4, fully_connected_layers=[1028])
mnist_args_ = dict(dim_h=64, batch_norm=True, f_size=5, pad=2, stride=2, min_dim=7, nonlinearity='LeakyReLU')
convnet_args_ = dict(dim_h=64, batch_norm=True, n_steps=3, nonlinearity='LeakyReLU')


def routine(data, models, losses, results, viz, key='classifier', criterion=None):
    classifier = models[key]
    inputs, targets = data.get_batch('images', 'targets')

    predicted = classify(classifier, inputs, targets, losses=losses, results=results, criterion=criterion, key=key)

    visualize(inputs, targets, predicted, viz=viz)


def classify(classifier, inputs, targets, losses=None, results=None, criterion=None, backprop_input=False,
             key='classifier'):
    criterion = criterion or nn.CrossEntropyLoss()

    #if not backprop_input:
    #    inputs = inputs.detach()

    outputs = classifier(inputs)
    predicted = torch.max(F.log_softmax(outputs, dim=1).data, 1)[1]

    if losses is not None:
        loss = criterion(outputs, targets)
        losses[key] = loss

    if results is not None:
        correct = 100. * predicted.eq(targets.data).cpu().sum() / targets.size(0)
        results[key + '_accuracy'] = correct

    return predicted


def visualize(viz_inputs, targets, predicted, viz=None, key='classifier'):
    if viz:
        viz.add_image(viz_inputs.data, labels=(targets.data, predicted.data), name=key + '_gt_pred')

# CORTEX ===============================================================================================================
# Must include `BUILD`, `TRAIN_ROUTINES`, and `DEFAULT_CONFIG`

def BUILD(data, models, model_type='convnet', dropout=0.2, classifier_args=None):
    classifier_args = classifier_args or {}
    shape = data.get_dims('x', 'y', 'c')
    dim_l = data.get_dims('labels')

    if model_type == 'resnet':
        from .modules.resnets import ResEncoder as Encoder
        args = resnet_args_
    elif model_type == 'convnet':
        from .modules.convnets import SimpleConvEncoder as Encoder
        args = convnet_args_
    elif model_type == 'mnist':
        from .modules.convnets import SimpleConvEncoder as Encoder
        args = mnist_args_
    else:
        raise NotImplementedError(model_type)

    args.update(**classifier_args)

    if shape[0] == 64:
        args['n_steps'] = 4

    classifier = Encoder(shape, dim_out=dim_l, dropout=dropout, **args)
    models.update(classifier=classifier)


TRAIN_ROUTINES = dict(classify=routine)
DEFAULT_CONFIG = dict(
    data=dict(batch_size=128),
    optimizer=dict(optimizer='Adam', learning_rate=1e-4),
    model=dict(dropout=0.2, model_type='convnet'),
    routines=dict(criterion=nn.CrossEntropyLoss()),
    train=dict(epochs=200, archive_every=10, save_on_best='losses.classifier')
)