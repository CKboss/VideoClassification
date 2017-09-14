import numpy as np
import pprint
import random
import os
import logging
import glob

import tensorflow as tf

import TFFusions.Config.Config as Config
from TFFusions.Train.load_yaml_to_FLAG import LOAD_YAML_TO_FLAG, Get_GlobalFLAG
from TFFusions.all_frame_models.frame_level_models import GetFrameModel
from TFFusions.Train.train_2 import split_into_small_peice,read_and_decode
from TFFusions.losses import SoftmaxLoss
from TFFusions.average_precision_calculator import mean_ap, accuracy
from TFFusions.Logger import Logger

# train_config = '/datacenter/1/LSVC/Code/VideoClassification/TrainScript/Server202/Eval_NetVLAD_EX3.yaml'
# train_config = '/datacenter/1/LSVC/Code/VideoClassification/TrainScript/Server202/Eval_GateDbof_VideoOnly_save16000_EX2.yaml'
# train_config = '/datacenter/1/LSVC/Code/VideoClassification/TrainScript/Server202/Eval_lstm_cell1024_EX20.yaml'
# train_config = '/datacenter/1/LSVC/Code/VideoClassification/TrainScript/Server202/Eval_GateDbof_VideoOnly_save92000.yaml'
# train_config = '/datacenter/1/LSVC/Code/VideoClassification/TrainScript/Server202/Eval_netFV_EX1.yaml'

# train_config = '/datacenter/1/LSVC/Code/VideoClassification/TrainScript/Server202/Eval_LstmATT_EX20_save78000_for_test.yaml'
# train_config = '/datacenter/1/LSVC/Code/VideoClassification/TrainScript/Server202/Eval_GateDbof_VideoOnly_save92000_test.yaml'
train_config = '/datacenter/1/LSVC/Code/VideoClassification/TrainScript/Server202/Eval_GateDbof_VideoOnly_save16000_EX2_test.yaml'

LOAD_YAML_TO_FLAG(train_config)
FLAGS = Get_GlobalFLAG()

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

    frame_len = tffeatures['frame_len']
    features = tf.decode_raw(tffeatures['features'], tf.float32)
    global FLAGS
    FEATURE_SIZE = getattr(FLAGS,'feature_size',1024)
    features = tf.reshape(features, [600, FEATURE_SIZE])
    labels = tf.decode_raw(tffeatures['labels'], tf.int32)
    labels = tf.reshape(labels, [500])
    name = tffeatures['name']
    frame_len_batch, features_batch, labels_batch, name_batch = tf.train.batch([frame_len, features, labels, name],
                                                                                       batch_size=batch_size,
                                                                                       num_threads=10,
                                                                                       capacity=128)
    return frame_len_batch, features_batch, labels_batch, name_batch


#
# def split_into_small_peice(features, target_label, video_frames, fix_lenght=10, scale=8, one_hot=True, only_shuffle=False):
#     '''
#     :param features:  a tensor batchsize x max_frams_len x features
#     :param target_label: a tensor batchsize x 500
#     :param video_frames: a tensor batchsize
#     :param fix_lenght: a int output features's len
#     :param scale: output batchsize will be mulit scale
#     :param only_shuffle: if set True will gen data without of order
#     :return:
#         features: (scale x batchsize) x fix_lenght x features
#         target_label: (scale x batchsize) x 500
#         videoframes: (scale x batchsize)
#     '''
#     global FLAGS
#
#     fix_lenght = getattr(FLAGS, 'fix_length', fix_lenght)
#     scale = getattr(FLAGS, 'scale', scale)
#     one_hot = getattr(FLAGS, 'one_hot', one_hot)
#     only_shuffle = getattr(FLAGS,'only_shuffle',only_shuffle)
#
#     n = features.shape[0]
#     n2 = target_label.shape[-1]
#     m = features.shape[-1]
#
#     features_ret = []
#     target_label_ret = []
#     video_frames_ret = []
#
#     if only_shuffle == False:
#         for i in range(n):
#             video_len = video_frames[i]
#             rid = random.choices(list(range(fix_lenght)), k=scale)
#
#             ff = features[i,:,:]
#             if video_len <= fix_lenght:
#                 ff = np.tile(ff,(fix_lenght//video_len+1,1))
#                 video_len = fix_lenght + 1
#
#             for rg in rid:
#                 l = rg
#                 r = rg + fix_lenght - 1
#                 if r >= video_len:
#                     l = video_len - fix_lenght - 1
#                     r = video_len - 2
#                 features_ret.append(ff[l:r + 1, :])
#                 video_frames_ret.append(fix_lenght)
#                 if one_hot == True:
#                     target_label_ret.append(target_label[i])
#                 else:
#                     target_label_ret.append(np.argmax(target_label[i]))
#     elif only_shuffle == True:
#
#         for i in range(n):
#             video_len = video_frames[i]
#             ff = features[i,:,:]
#             if video_len <= fix_lenght:
#                 ff = np.tile(ff,(fix_lenght//video_len+1,1))
#                 video_len = fix_lenght + 1
#             for j in range(scale):
#                 ids = random.choices(list(range(video_len)), k=fix_lenght)
#                 for k in ids:
#                     features_ret.append(ff[k, :])
#                 video_frames_ret.append(fix_lenght)
#                 if one_hot == True:
#                     target_label_ret.append(target_label[i])
#                 else:
#                     target_label_ret.append(np.argmax(target_label[i]))
#
#     features_ret = np.vstack(features_ret).reshape(-1, fix_lenght, m)
#     video_frames_ret = np.array(video_frames_ret)
#
#     if one_hot == True:
#         target_label_ret = np.vstack(target_label_ret).reshape(-1, n2)
#     else:
#         target_label_ret = np.array(target_label_ret)
#
#     return features_ret, target_label_ret, video_frames_ret


if os.path.exists(FLAGS.train_dir) == False:
    print('mk train dir {}'.format(FLAGS.train_dir))
    os.mkdir(FLAGS.train_dir)
batchsize = FLAGS.batchsize

one_hot = getattr(FLAGS, 'one_hot', False)

# model
if FLAGS.device_id != None:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.device_id)[1:-1]

FEATURE_SIZE = getattr(FLAGS,'feature_size',1024)
inputs = tf.placeholder(dtype=tf.float32, shape=(batchsize * FLAGS.scale, FLAGS.fix_length, FEATURE_SIZE))
num_frames = tf.placeholder(dtype=tf.int32, shape=(batchsize * FLAGS.scale))

if one_hot == True:
    target_labels = tf.placeholder(dtype=tf.int32, shape=(batchsize * FLAGS.scale, FLAGS.vocab_size))
else:
    target_labels = tf.placeholder(dtype=tf.int32, shape=(batchsize * FLAGS.scale))

model = GetFrameModel(FLAGS.frame_level_model)()

predict_labels = model.create_model(model_input=inputs, vocab_size=FLAGS.vocab_size, num_frames=num_frames,
                                    num_mixtures=FLAGS.moe_num_mixtures)
predict_labels = predict_labels['predictions']

enable_softmax = getattr(FLAGS,'enable_softmax',False)
if enable_softmax:
    predict_labels = tf.nn.softmax(predict_labels)

tp_1 = tf.nn.top_k(predict_labels,k=1)
tp_5 = tf.nn.top_k(predict_labels,k=5)
tp_10 = tf.nn.top_k(predict_labels,k=10)

# init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
init_op = tf.global_variables_initializer()

# Load from TFRecord
data_kind = getattr(FLAGS,'train_data','inc')


if getattr(FLAGS,'is_test',False) == True:
    suffix_ = 'test_*'
else :
    suffix_ = 'val_*'

print('eval for {}'.format(suffix_))

if data_kind == 'inc':
    val_file_list = glob.glob('/datacenter/1/LSVC/inc_tfrecords/'+suffix_)
elif data_kind == 'vgg':
    val_file_list = glob.glob('/datacenter/1/LSVC/tfrecords/'+suffix_)
elif data_kind == 'sen':
    val_file_list = glob.glob('/datacenter/1/LSVC/sen_tfrecords/'+suffix_)


val_file_queue = tf.train.string_input_producer(val_file_list)
test_frame_len_batch, test_feature_batch, test_label_batch, test_name_batch = read_and_decode(val_file_queue, FLAGS.batchsize)

# LOG
log_prefix_name = 'Eval_{}_{}'.format(FLAGS.name, FLAGS.EX_ID)
# python's logging
pylog = logging.getLogger(log_prefix_name)
pylog.setLevel(logging.DEBUG)
fh = logging.FileHandler(FLAGS.train_dir + '/' + log_prefix_name + '.log')
ch = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s-%(name)s-%(levelname)s: %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
pylog.addHandler(fh)
pylog.addHandler(ch)

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf_config.allow_soft_placement = True
tf_config.log_device_placement = True

# init session
sess = tf.Session(config=tf_config)
# sess.run(init_op)

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord, sess=sess)

# save ( after session )
Saver = tf.train.Saver(max_to_keep=20, keep_checkpoint_every_n_hours=2)

if FLAGS.model_checkpoint_path is not None:
    print('load model from {} ...'.format(FLAGS.model_checkpoint_path))
    Saver.restore(sess=sess, save_path=FLAGS.model_checkpoint_path)
    print('Success !!!')
else:
    raise NotImplementedError

pylog.info('train_config: {}'.format(FLAGS.YAML))

cnt = 0

if suffix_ == 'test_*':
    loop = getattr(FLAGS,'loop',80000//getattr(FLAGS,'batchsize',32))
elif suffix_ == 'val_*':
    loop = getattr(FLAGS,'loop',16000//getattr(FLAGS,'batchsize',32))

print('loop:',loop)

acc_1 = np.zeros((500))
acc_5 = np.zeros((500))
acc_10 = np.zeros((500))
label_cnt = np.zeros(500)
predict_result = []
correct_labels = []
video_names = []

for i in range(loop):

    if i % 20 == 0:
        print('loop {} ....'.format(i))

    input_features, input_target_labels, input_video_frames, test_name = sess.run(
        [test_feature_batch, test_label_batch, test_frame_len_batch, test_name_batch])
    input_features, input_target_labels, input_video_frames = split_into_small_peice(input_features, input_target_labels, input_video_frames)
    fd = {inputs: input_features, target_labels: input_target_labels, num_frames: input_video_frames}

    (_,tf_top_10),pred = sess.run([tp_10,predict_labels],feed_dict=fd)

    Labels = np.argmax(input_target_labels,axis=1)

    video_names.append(test_name)

    for j in range(FLAGS.batchsize):

        l = j*FLAGS.scale
        r = l+FLAGS.scale
        ped = np.sum(pred[l:r],axis=0)/FLAGS.scale
        top_10 = np.argsort(ped)[-10:][::-1]
        label = Labels[l]

        if label == top_10[0]:
            acc_1[label] += 1
        if label in top_10[:5]:
            acc_5[label] += 1
        if label in top_10:
            acc_10[label] += 1

        label_cnt[label] += 1

        predict_result.append(np.array(ped))
        correct_labels.append(label)

predict_result = np.concatenate(predict_result).reshape(-1,500)
correct_labels = np.array(correct_labels)
video_names = np.concatenate(video_names)

file = FLAGS.train_dir+FLAGS.save_file
np.savez(file,acc_1=acc_1,acc_5=acc_5,acc_10=acc_10,
         label_cnt=label_cnt,predict_result=predict_result,correct_labels=correct_labels,video_names=video_names)

coord.request_stop()
coord.join(threads)

