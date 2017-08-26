import TFFusions.Config.Config as Config
from TFFusions.toolkits.dataloader import getTrainItems,concurrent_get_items

import tensorflow as tf
import numpy as np

def _bytes_feature(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value=[value]))

def _int64_feature(value):
    if isinstance(value,list) == False:
        return tf.train.Feature(int64_list = tf.train.Int64List(value=[value]))
    else:
        return tf.train.Feature(int64_list = tf.train.Int64List(value=value))


# make tfrecord
def make_tfrecord(recordfilename):

    trainitems = getTrainItems()
    trainitems = trainitems[:64]

    writer = tf.python_io.TFRecordWriter(recordfilename)

    cnt = 100
    for item in trainitems:
        frame_len,features,labellst = concurrent_get_items(item,kind='train')
        features = np.pad(features,((0,600-frame_len),(0,0)),'constant')
        labels = np.zeros(500,dtype=np.int32)
        for label in labellst: labels[label] = 1

        example = tf.train.Example(features=tf.train.Features(feature={
            'cnt': _int64_feature(cnt),
            'frame_len': _int64_feature(frame_len),
            'features': _bytes_feature(features.tostring()),
            'labels': _bytes_feature(labels.tostring())
        }))

        cnt+=1

        writer.write(example.SerializeToString())

    writer.close()

recordfilename = '/tmp/tf2.records'

make_tfrecord(recordfilename)

def read_and_decode(filename_queue,batch_size):

    reader = tf.TFRecordReader()

    _,serialized_example = reader.read(filename_queue)

    tffeatures = tf.parse_single_example(serialized_example,
        features={
            'cnt': tf.FixedLenFeature([],tf.int64),
            'frame_len': tf.FixedLenFeature([],tf.int64),
            'features': tf.FixedLenFeature([],tf.string),
            'labels': tf.FixedLenFeature([],tf.string),
        })

    frame_len = tffeatures['frame_len']
    features = tf.decode_raw(tffeatures['features'],tf.float32)
    features = tf.reshape(features,[600,4096])
    labels = tf.decode_raw(tffeatures['labels'],tf.int32)
    labels = tf.reshape(labels,[500])
    id = tffeatures['cnt']


    frame_len_batch, features_batch, labels_batch,id_batch = tf.train.batch([frame_len,features,labels,id],
                                                                     batch_size = batch_size )
    return frame_len_batch,features_batch,labels_batch,id_batch

filenamequeue = tf.train.string_input_producer(['/tmp/tf1.records','/tmp/tf2.records'])

a,b,c,d = read_and_decode(filenamequeue,10)
a1,b1,c1,d1 = read_and_decode(filenamequeue,10)

init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())

with tf.Session() as sess:

    sess.run(init_op)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord,sess=sess)

    for i in range(90):
        A,B,C,D = sess.run([a,b,c,d])
        print('-->',D)
        A,B,C,D = sess.run([a1,b1,c1,d1])

    coord.request_stop()
    coord.join(threads)





