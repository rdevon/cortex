
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
In case you do not have any of these, you can install them simply using pip.
# About the data
The data consists of over 1000 preprocessed functional data samples. The dimensions of the volumes are normalized but different individual scans may have different number of timesteps. Hence a tfrecord feature will be a pair of the image data and its shape. Here is a code snippet to make this clear:
```
example = tf.train.Example(features=tf.train.Features(feature={'img_raw': _bytes_feature(img_raw), 'img_shape': _bytes_feature(img_shape_raw)}))
```
The data is minimally preprocessed using slice timing correction, realignment to correct for motion and written into template space at 3x3x3 mm<sup>3</sup> isotropic resolution. Also the datatype is float64.

# Conversion
The script lets the user decide, the number of files to include in the tfrecord file. It then downloads the files and starts adding them into the tfrecord file. Once all the files are added, you will have your tfrecord file in the home directory of your OS. Just run the code and follow the runtime instructions.

# Significance
Functional images of brain are bulky and reading through all the training samples can get inefficient and time consuming. Converting this data into tfrecords makes it ready for tensorflow and accessing and batching data becomes better and easy.




[1]: http://www.frontiersin.org/10.3389/conf.fninf.2013.09.00041/event_abstract "Cameron Craddock, Yassine Benhajali, Carlton Chu, Francois Chouinard, Alan Evans, András Jakab, Budhachandra Singh Khundrakpam, John David Lewis, Qingyang Li, Michael Milham, Chaogan Yan, Pierre Bellec (2013). The Neuro Bureau Preprocessing Initiative: open sharing of preprocessed neuroimaging data and derivatives. In Neuroinformatics 2013, Stockholm, Sweden."
[2]: http://preprocessed-connectomes-project.org/abide/index.html "Know more about this dataset"
