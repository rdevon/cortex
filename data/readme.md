
---
# Converting fmri dataset into tfrecord file
---

# Introduction
This is a tutorial on converting resting state fmri data from the dataset called ABIDE (Autism Brain Imaging Data Exchange)[1], into more accessible tfrecords file format. A consortium of the International Neuroimaging Datasharing Initiative (INDI), ABIDE is a collaboration of 16 international imaging sites that have aggregated and are openly sharing neuroimaging data from 539 individuals suffering from ASD and 573 typical controls.[2]
# Dependencies
Only dependencies to run this code are the python libraries imported. These are the imports:
```
import os
import numpy as np
import nibabel as nib
import tensorflow as tf
import urllib2
import pandas as pd
```
[1]: http://www.frontiersin.org/10.3389/conf.fninf.2013.09.00041/event_abstract "Cameron Craddock, Yassine Benhajali, Carlton Chu, Francois Chouinard, Alan Evans, András Jakab, Budhachandra Singh Khundrakpam, John David Lewis, Qingyang Li, Michael Milham, Chaogan Yan, Pierre Bellec (2013). The Neuro Bureau Preprocessing Initiative: open sharing of preprocessed neuroimaging data and derivatives. In Neuroinformatics 2013, Stockholm, Sweden."
[2]: http://preprocessed-connectomes-project.org/abide/index.html "Know more about this dataset"
