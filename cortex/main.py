'''Main file for running experiments.

'''


import logging

from cortex._lib import exp
from cortex._lib.data import setup as setup_data
from cortex._lib.models import build_networks
from cortex._lib.optimizer import setup as setup_optimizer
from cortex._lib.train import setup as setup_train, main_loop
from cortex._lib.utils import print_section

import yaml
from tkinter import filedialog
from os.path import expanduser


__author__ = 'R Devon Hjelm'
__author_email__ = 'erroneus@gmail.com'


logger = logging.getLogger('cortex')
homeDirectory = expanduser("~")
configDirectory = homeDirectory + '/config.yml'


def configure():
    torchvisionDataPath = filedialog.askdirectory(title="Select torchvision data path.")
    dataPathsLoopSize = input("Please enter the number of data paths you want to add: ")
    dataPaths = dict()
    for x in range(int(dataPathsLoopSize)):
        datasetName = input("Please enter dataset {} name (i.e.: CelebA): ".format(x + 1))
        dataPath = filedialog.askdirectory(title="Select data path {}.".format(x + 1))
        dataPaths[datasetName] = dataPath

    vizFontFilePath = filedialog.askopenfilename(title="Select viz/font file.")
    vizServerAdress = input("Please enter the viz server adress: ")
    vizConfig = dict(font=vizFontFilePath, server=vizServerAdress)
    outPath = filedialog.askdirectory(title="Select output path.")

    config = dict(
        torchvision_data_path=torchvisionDataPath,
        data_paths=dataPaths,
        viz=vizConfig,
        out_path=outPath
    )
    with open(configDirectory, 'w') as outfile:
        yaml.dump(config, outfile)


def main():
    '''Main function.

    '''
    # Parse the command-line arguments

    try:
        print_section('LOADING DATA')
        setup_data(**exp.ARGS.data)

        print_section('MODEL')
        build_networks(**exp.ARGS.builds)

        print_section('OPTIMIZER')
        setup_optimizer(**exp.ARGS.optimizer)

        print_section('TRAIN')
        setup_train()

    except KeyboardInterrupt:
        print('Cancelled')
        exit(0)

    main_loop(**exp.ARGS.train)
