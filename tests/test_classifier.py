from cortex._lib import (config, setup_experiment, exp)
from cortex.built_ins.models.utils import update_encoder_args
from cortex.built_ins.models.classifier import ImageClassification, SimpleClassifier
from .args_mock import args
import logging
from cortex._lib import (config, data, exp, optimizer, setup_cortex,
                         setup_experiment, train)
from cortex._lib.utils import print_section
import torch
import subprocess
import os


# def test_classifier_build():
#     classifier = SimpleClassifier()
#     print(dir(classifier))
#     assert 1 == 0
#     # print(classifier)
#     # def build(self, dim_in: int=None, classifier_args=dict(dim_h=[200, 200])):
#
#
