'''Main file for running experiments.

'''

import logging

'''
import lib.data
from lib import setup
from lib.data import setup_data
from lib import exp, setup_reload
from lib.exp import load_tensors, set_models
from lib.gen import setup_generation
from lib.loss import get_gan_loss
from lib.train import setup_optimizer, train
from lib.utils import print_section
from models import build, make_iterator_generator
'''


logger = logging.getLogger('cortex')



import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms



def main(eval_mode=False):
    '''Main function for continuous BGAN.

    '''
    data_args = exp.ARGS['data']
    model_args = exp.ARGS['model']
    loss_args = exp.ARGS['loss']
    optimizer_args = exp.ARGS['optimizer']
    train_args = exp.ARGS['train']

    print_section('LOADING DATA') ##############################################
    setup_data(make_iterator_generator, model_args['dim_z'], **data_args)

    print_section('MODEL') #####################################################
    noise = T.matrix('noise')
    features = T.tensor4(lib.data.IMAGE_VAR)
    logger.info('Building model and compiling GAN functions...')
    logger.info('Model args: {}'.format(model_args))
    inps = {'noise': noise, lib.data.IMAGE_VAR: features}
    exp.ITERATORS['train'](None, make_pbar=False).next()
    models, inputs = build(
        inps, data_iterator=exp.ITERATORS['train'](None, make_pbar=False),
        **model_args)
    set_models(models)

    print_section('LOSS') ######################################################

    losses, stats, samples, histograms, tensors = get_gan_loss(
        inputs, optimizer_args=optimizer_args, **loss_args)
    load_tensors(inputs, losses, stats, samples, histograms, tensors)

    print_section('OPTIMIZER') #################################################
    setup_optimizer(updates={}, **optimizer_args)
    setup_generation()

    if eval_mode:
        return
    print_section('TRAIN') #####################################################
    train(**train_args)


def reload_model(model_file):
    setup_reload(model_file)
    main(eval_mode=True)


_default_args = dict(
    data=dict(
        batch_size=64
    ),
    optimizer=dict(
        optimizer='adam',
        optimizer_options=dict(beta1=0.5),
        learning_rate=1e-4,
    ),
    model=dict(
        arch='resnet',
        dim_z=128,
        nonlin='tanh'
    ),
    loss=dict(
        loss='gan',
        penalty='gradient_norm',
        penalty_amount=1.0,
    ),
    train=dict(
        epochs=200,
        summary_updates=100,
        archive_every=1,
        updates_per_model=dict(discriminator=1, generator=1)
    )
)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


def train_model():
    use_cuda = torch.cuda.is_available()
    trainset = torchvision.datasets.MNIST(root='/home/devon/Data/MNIST', train=True, download=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    trainset = torchvision.datasets.MNIST(root='/home/devon/Data/MNIST', train=False, download=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=False, num_workers=2)

    net = Net()

    if use_cuda:
        net.cuda()
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    def train(epoch):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            optimizer.zero_grad()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.data[0]
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            #progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #    % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    def test(epoch):
        global best_acc
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs, volatile=True), Variable(targets)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.data[0]
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            #progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #             % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    acc = 100. * correct / total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.module if use_cuda else net,
            'acc': acc,
            'epoch': epoch,
        }
        #if not os.path.isdir('checkpoint'):
        #    os.mkdir('checkpoint')
        #torch.save(state, './checkpoint/ckpt.t7')
        best_acc = acc


    for epoch in range(start_epoch, start_epoch + 200):
        train(epoch)
        test(epoch)


if __name__ == '__main__':
    train_model()
    try:
        setup(_default_args)
        main()
    except KeyboardInterrupt:
        print 'Cancelled'
        exit(0)