'''
This script is for loading float data from a text file.  The only difference between float
and int reading is to use 'IntList64' and 'int64_list' instead of 'FloatList' and 'float_list'

command line arguments accept at most, 2 arguments.  The file names for the data and labels file

numpy is the most effective way, in my experience, to read in numeric data.  Tensorflow does have
a 'TextLineReader()' which allows for text reading as strings.  This would be useful if the data
was in byte form (or text).  However, numpy is the most streamlined for numeric data
'''

import tensorflow as tf
import numpy as np
import sys
import argparse
import os

def build_sample_file(base_name,x_size,y_size):
	if not os.path.isfile(base_name+'_data.txt'):
		D = np.random.rand(y_size,x_size)
		np.savetxt(base_name+'_data.txt',D,delimiter=',')
	if not os.path.isfile(base_name+'_labels.txt'):
		L = np.random.randint(2,size=y_size)
		np.savetxt(base_name+'_labels.txt',L,delimiter=',')
	return base_name+'_data.txt',base_name+'_labels.txt'

def load_file(name,delimiter=',',type=np.float32):
    return np.loadtxt(name,delimiter=delimiter,dtype=type)

def build_records(data):
    return tf.train.Feature(float_list = tf.train.FloatList(value=data))

#default file names
data_file = "./data_sample.txt"
labels_file = "./labels_sample.txt"

if __name__ == "__main__":
    delete = False

    parser = argparse.ArgumentParser(description='Run tfrecord loader example')
    parser.add_argument('--data', metavar='--D', help = 'The data input file location',default=None)
    parser.add_argument('--labels', metavar='--L',help = 'Label file location',default=None)
    parser.add_argument('--output', metavar='--O', help = 'Output directory',default='./tmp')

    args = parser.parse_args()

    print args.data
    print args.labels
    print args.output

    if args.data == None or args.labels == None:
        data_file,labels_file = build_sample_file('test', 10, 1500)
        print "Missing input file.  Will use randomly generated data"
        delete = True
    else:
        data_file = args.data
        labels_file = args.labels

    if not os.path.isdir(args.output):
        os.makedirs(args.output)
    record_name = "text_records.tfrecords"
    record_path_name = args.output+'/'+record_name
    features = load_file(data_file)
    labels = load_file(labels_file)
    if delete:
        os.remove(data_file)
        os.remove(labels_file)

    record_writer = tf.python_io.TFRecordWriter(record_path_name)

    for d, l in zip(features,labels):

        data_record = build_records(d)
        #NOTE: The value argument must be iterable.  In this case, l is a single float.  Which would
        #not be the case if the data was one_hot
        labels_record = build_records([l])

        #each feature can be of different sizes.  However, in this case, that is not required
        output = tf.train.Example(features=tf.train.Features(
            feature={'data': data_record, 'labels': labels_record}))
        record_writer.write(output.SerializeToString())

    print "Save completed.  File is: "+record_path_name
    record_writer.close()






