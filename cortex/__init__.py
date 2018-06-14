'''Init file for cortex.

'''

from cortex._lib import config, setup_cortex
config.set_config()  # Dataset plugins rely on paths.

from cortex.built_ins.datasets import *
from cortex.built_ins.models import *


setup_cortex()
