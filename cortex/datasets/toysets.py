# -*- coding: utf-8 -*-
r"""
:mod:`cortex.core.datasets` -- Small toy datasets for interpretable experimentation
==================================================================================

.. module:: datasets
   :platform: Unix
   :synopsis: Basically everything found in this website
      <https://cs.joensuu.fi/sipu/datasets/>_ (for now).

Collection of datasets (mostly 2D) used primarily for benchmarking of
inference algorithms and interpretable experiments in the input space.

TODOs
-----
1. Include common datasets for toying with GANs, like the balanced 2-moons
2. Fix module title once a proper packaging scheme is introduced

"""
__author__ = 'Tsirigotis Christos'
__author_email__ = 'tsirif@gmail.com'

import errno
import itertools as it
import os

import torch
import torch.utils.data as data


DATASETS = ["G2", "S_set", "A_set", "DIM_set", "Unbalance",
            "Aggregation", "Compound", "Pathbased", "Spiral",
            "D31", "R15", "Jain", "Flame"]
DIM_VARIANT_DATASETS = ["G2", "DIM_set"]
SD_VARIANT_DATASETS = ["G2"]
NUM_VARIANT_DATASETS = ["S_set", "A_set"]

