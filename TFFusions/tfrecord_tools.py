import TFFusions.Config.Config as Config
from TFFusions.toolkits.dataloader import getTrainItems, concurrent_get_items, getTestItems, getValItems, Load_Features_SENET, Load_Features_INC
from TFFusions.Train.load_yaml_to_FLAG import Get_GlobalFLAG

import tensorflow as tf
import numpy as np


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    if isinstance(value, list) == False:
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    else:
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


# make tfrecord
def make_tfrecord(items, filename, kind):
    writer = tf.python_io.TFRecordWriter(filename)
    cnt = 0

    for item in items:
        name = item[0]
        try:
            frame_len, features, labellst = concurrent_get_items(item, kind=kind, load_func=Load_Features_SENET)
            assert frame_len != 0
        except Exception as E:
            print(E)
            continue

        features = np.pad(features, ((0, 600 - frame_len), (0, 0)), 'constant')
        labels = np.zeros(500, dtype=np.int32)
        for label in labellst: labels[label] = 1

        example = tf.train.Example(features=tf.train.Features(feature={
            'name': _bytes_feature(name.encode()),
            'frame_len': _int64_feature(frame_len),
            'features': _bytes_feature(features.tostring()),
            'labels': _bytes_feature(labels.tostring())
        }))

        writer.write(example.SerializeToString())

        cnt += 1
        if cnt % 100 == 0:
            print('cnt: {} ...'.format(cnt))

    writer.close()


def RUN_make_TF_records():

    # valitems = getValItems()
    # trainitems = getTrainItems()
    testitems = getTestItems()

    # n = len(valitems)
    # n = len(trainitems)
    n = len(testitems)
    duansize = 10240
    duan = n // duansize + 1

    # prefixname = '/mnt/md0/LSVC/tfrecords/train_tf_{}_{}.tfrecord'
    # prefixname = '/mnt/md0/LSVC/inc_tfrecords/val_tf_{}_{}.tfrecord'
    prefixname = '/datacenter/1/LSVC/sen_tfrecords/test_tf_{}_{}.tfrecord'

    for i in range(duan):
        l = i * duansize
        r = min(l + duansize, n - 1)
        filename = prefixname.format(l, r - 1)
        # items = valitems[l:r]
        # items = trainitems[l:r]
        items = testitems[l:r]
        print(filename + '....')
        make_tfrecord(items, filename, kind='test')
        # make_tfrecord(items, filename, kind='train')

FLAGS = None

def read_and_decode(filename_queue, batch_size):
    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)

    tffeatures = tf.parse_single_example(serialized_example,
                                         features={
                                             'name': tf.FixedLenFeature([], tf.string),
                                             'frame_len': tf.FixedLenFeature([], tf.int64),
                                             'features': tf.FixedLenFeature([], tf.string),
                                             'labels': tf.FixedLenFeature([], tf.string),
                                         })
    global FLAGS
    if FLAGS is None:
        FLAGS = Get_GlobalFLAG()
    FEATURE_SIZE = getattr(FLAGS,'feature_size',1024)

    frame_len = tffeatures['frame_len']
    features = tf.decode_raw(tffeatures['features'], tf.float32)
    features = tf.reshape(features, [600, FEATURE_SIZE])
    labels = tf.decode_raw(tffeatures['labels'], tf.int32)
    labels = tf.reshape(labels, [500])
    name = tffeatures['name']

    frame_len_batch, features_batch, labels_batch, name_batch = tf.train.shuffle_batch([frame_len, features, labels, name],
                                                                                   min_after_dequeue=32,
                                                                                   batch_size=batch_size,
                                                                                   num_threads=10,
                                                                                   capacity=128)
    return frame_len_batch, features_batch, labels_batch, name_batch


def test():

    kind = 'train'
    items = getTrainItems()
    item = items[0]

    filenamequeue = tf.train.string_input_producer(['/mnt/md0/LSVC/inc_tfrecords/val_tf_0_10239.tfrecord'])

    a, b, c, d = read_and_decode(filenamequeue, 10)
    a1, b1, c1, d1 = read_and_decode(filenamequeue, 10)

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    with tf.Session() as sess:
        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

        for i in range(20):
            A, B, C, D = sess.run([a, b, c, d])
            print('-->', D)
            # A,B,C,D = sess.run([a1,b1,c1,d1])

        coord.request_stop()
        coord.join(threads)
