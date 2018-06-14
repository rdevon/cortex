
import yaml
from tkinter import filedialog
from os.path import expanduser

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
