import tensorflow as tf
import numpy as np
import sys

'''
This script is for loading float data from a text file.  The only difference between float
and int reading is to use 'IntList64' and 'int64_list' instead of 'FloatList' and 'float_list'

command line arguments accept at most, 2 arguments.  The file names for the data and labels file
'''

'''
numpy is the most effective way, in my experience, to read in numeric data.  Tensorflow does have
a 'TextLineReader()' which allows for text reading as strings.  This would be useful if the data
was in byte form (or text).  However, numpy is the most streamlined for numeric data
'''
def load_file(name,delimiter=',',type=np.float32):
    return np.loadtxt(name,delimiter=delimiter,dtype=type)


def build_records(data):
    return tf.train.Feature(float_list = tf.train.FloatList(value=data))

#default file names
data_file = "./data_sample.txt"
labels_file = "./labels_sample.txt"

if __name__ == "__main__":
    args = sys.argv

    if len(args) == 2:
        data_file = args[0]
        labels_file = args[0]

    record_name = "./text_records.tfrecords"
    features = load_file(data_file)
    labels = load_file(labels_file)
    record_writer = tf.python_io.TFRecordWriter(record_name)

    for d, l in zip(features,labels):

        data_record = build_records(d)
        #NOTE: The value argument must be iterable.  In this case, l is a single float.  Which would
        #not be the case if the data was one_hot
        labels_record = build_records([l])

        #each feature can be of different sizes.  However, in this case, that is not required
        output = tf.train.Example(features=tf.train.Features(
            feature={'data': data_record, 'labels': labels_record}))
        record_writer.write(output.SerializeToString())

    print "Save completed.  File is: ./text_records.tfrecords"
    record_writer.close()






