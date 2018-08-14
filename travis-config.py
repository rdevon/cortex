import yaml
from os import path

config_name = '.cortex.yml'
home = path.expanduser('~')
config_file_path = path.join(home, config_name)

configs = dict(
    datapaths=dict(local=home, torchvision=home),
    out_path=home,
    viz=dict(port=8097, server='http://localhost'))

with open(config_file_path, 'w') as config_file:
    yaml.dump(configs, config_file)
