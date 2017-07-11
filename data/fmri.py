import os
import numpy as np
import nibabel as nib
import tensorflow as tf
import urllib2
import pandas as pd


#Fetching and processing list of names of files
f = urllib2.urlopen('https://storage.googleapis.com/sampledatamri/Phenotypic_V1_0b_preprocessed1.csv')
with open("metadata.csv", "wb") as code:
	code.write(f.read())
f.close()
df = pd.read_csv('metadata.csv', skipinitialspace=True, usecols = ['FILE_ID'])


# list of filenames
names = df.FILE_ID.tolist()
n_files = len(names)
os.remove("metadata.csv")


# take input from user
print "\nTotal files in the database: \t" + str(n_files) + "\t(of size ~200MB each)"
print "How many files would you like to include in your tf.records file: \t"
while True:
	inp = raw_input('-> ')
	try:
		inp = int(inp)
	except ValueError:
		print "enter a valid number"
		continue
	if 0 < inp <= n_files:
		break;
	else:
		print "enter a valid number"


#some functions for tfrecords
def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
	

#some variables for tfrecords
tfrecords_filename = 'data.tfrecords'
writer = tf.python_io.TFRecordWriter(tfrecords_filename)


#download data and convert to tfrecords file
url1 = "https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Outputs/cpac/func_minimal/"
ext = "_func_minimal.nii.gz"
print '\n'
for i in range(inp):
	fil = str(names[i])
	url = url1 + fil + ext
	file_name = url.split('/')[-1]
	u = urllib2.urlopen(url)
	f = open(file_name, 'wb')
	meta = u.info()
	file_size = int(meta.getheaders("Content-Length")[0])
	print "Downloading: %s Bytes: %s" % (file_name, file_size)
	file_size_dl = 0
	block_sz = 8192
	while True:
		buffer = u.read(block_sz)
		if not buffer:
			break

		file_size_dl += len(buffer)
		f.write(buffer)
		status = r"%10d  [%3.2f%%]" % (file_size_dl, file_size_dl * 100. / file_size)
		status = status + chr(8)*(len(status)+1)
		print status,
	f.close()

	img = nib.load(file_name)
	img_data = img.get_data()
	img_shape = img_data.shape
	img_raw = img_data.tostring()
	img_shape_raw = str(img_shape)
	example = tf.train.Example(features=tf.train.Features(feature={'img_raw': _bytes_feature(img_raw), 'img_shape': _bytes_feature(img_shape_raw)}))
	writer.write(example.SerializeToString())
	print '\nfile\t' + str(i+1) + '\tadded to tfrecords file'
	os.remove(file_name)
writer.close()
print '\nThe file "data.tfrecords" is saved in your home directory\n'
