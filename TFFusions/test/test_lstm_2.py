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
from TFFusions.losses import SoftmaxLoss
from TFFusions.average_precision_calculator import mean_ap, accuracy
from TFFusions.Logger import Logger
from TFFusions.tfrecord_tools import read_and_decode

FLAGS = None

def find_class_by_name(name, models):
    classes = [getattr(model, name, None) for model in models]
    if len(classes) == 1:
        return classes[0]
    else:
        return classes

#############################################################################

def split_into_small_peice(features, target_label, video_frames, fix_lenght=10, scale=8, one_hot=True):
    '''
    :param features:  a tensor batchsize x max_frams_len x features
    :param target_label: a tensor batchsize x 500
    :param video_frames: a tensor batchsize
    :param fix_lenght: a int output features's len
    :param scale: output batchsize will be mulit scale
    :return:
        features: (scale x batchsize) x fix_lenght x features
        target_label: (scale x batchsize) x 500
        videoframes: (scale x batchsize)
    '''
    global FLAGS

    fix_lenght = getattr(FLAGS, 'fix_length', fix_lenght)
    scale = getattr(FLAGS, 'scale', scale)
    one_hot = getattr(FLAGS, 'one_hot', one_hot)

    n = features.shape[0]
    n2 = target_label.shape[-1]
    m = features.shape[-1]

    features_ret = []
    target_label_ret = []
    video_frames_ret = []

    for i in range(n):
        video_len = video_frames[i]
        rid = random.choices(list(range(fix_lenght)), k=scale)

        ff = features[i,:,:]
        if video_len <= fix_lenght:
            ff = np.tile(ff,(fix_lenght//video_len+1,1))
            video_len = fix_lenght + 1

        for rg in rid:
            l = rg
            r = rg + fix_lenght - 1
            if r >= video_len:
                l = video_len - fix_lenght - 1
                r = video_len - 2
            features_ret.append(ff[l:r + 1, :])
            video_frames_ret.append(fix_lenght)
            if one_hot == True:
                target_label_ret.append(target_label[i])
            else:
                target_label_ret.append(np.argmax(target_label[i]))

    features_ret = np.vstack(features_ret).reshape(-1, fix_lenght, m)
    video_frames_ret = np.array(video_frames_ret)

    if one_hot == True:
        target_label_ret = np.vstack(target_label_ret).reshape(-1, n2)
    else:
        target_label_ret = np.array(target_label_ret)

    return features_ret, target_label_ret, video_frames_ret


def toOneHot(x, vocab=500):
    '''
    :param x: a ndarray (batch)
    :param vocab: classes num
    :return: a one_hot array (batch x vocab)
    '''
    batchsize = x.shape[0]
    ret = np.zeros(shape=(batchsize, vocab))
    for i in range(batchsize):
        ret[i, x[i]] = 1
    return ret


#############################################################################


train_config = '/datacenter/1/LSVC/Code/VideoClassification/TrainScript/Server209/LstmAttentionLstmModel_EX19_2.yaml'
LOAD_YAML_TO_FLAG(train_config)
FLAGS = Get_GlobalFLAG()

FLAGS.train_dir = '/tmp/test/'
if os.path.exists(FLAGS.train_dir) == False:
    print('mk train dir {}'.format(FLAGS.train_dir))
    os.mkdir(FLAGS.train_dir)
batchsize = FLAGS.batchsize

one_hot = getattr(FLAGS, 'one_hot', False)

# model
if FLAGS.device_id != None:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.device_id)[1:-1]

inputs = tf.placeholder(dtype=tf.float32, shape=(batchsize * FLAGS.scale, FLAGS.fix_length, 1024))
num_frames = tf.placeholder(dtype=tf.int32, shape=(batchsize * FLAGS.scale))

if one_hot == True:
    target_labels = tf.placeholder(dtype=tf.int32, shape=(batchsize * FLAGS.scale, FLAGS.vocab_size))
else:
    target_labels = tf.placeholder(dtype=tf.int32, shape=(batchsize * FLAGS.scale))

model = GetFrameModel(FLAGS.frame_level_model)()

predict_labels = model.create_model(model_input=inputs, vocab_size=FLAGS.vocab_size, num_frames=num_frames,
                                    num_mixtures=FLAGS.moe_num_mixtures)
predict_labels = predict_labels['predictions']

vars = tf.trainable_variables()
lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars if 'bias' not in v.name]) * FLAGS.regularization_penalty

# loss
if one_hot == True:
    lossfunc = SoftmaxLoss()
    loss = lossfunc.calculate_loss(predict_labels, target_labels) + lossL2
else:
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=predict_labels, labels=target_labels)
    loss = tf.reduce_mean(loss) + lossL2

# optimize
global_step = tf.Variable(0, trainable=False)
decayed_learning_rate = tf.train.exponential_decay(FLAGS.base_learning_rate,
                                                   global_step,
                                                   FLAGS.decay_at_step,
                                                   FLAGS.learning_rate_decay,
                                                   staircase=True)
optimizer_class = find_class_by_name(FLAGS.optimize, [tf.train])
# train_op = tf.train.GradientDescentOptimizer(learning_rate=decayed_learning_rate).minimize(loss,global_step=global_step)
train_op = optimizer_class(learning_rate=decayed_learning_rate).minimize(loss, global_step=global_step)

# init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
init_op = tf.global_variables_initializer()

# Load from TFRecord
# val_file_list = glob.glob('/mnt/md0/LSVC/inc_tfrecords/val_*')
train_file_list = glob.glob('/datacenter/1/LSVC/inc_tfrecords/train_*')

train_file_queue = tf.train.string_input_producer(train_file_list)
# val_file_queue = tf.train.string_input_producer(val_file_list)
train_frame_len_batch, train_feature_batch, train_label_batch, train_name_batch = read_and_decode(train_file_queue,
                                                                                                  FLAGS.batchsize)
# test_frame_len_batch, test_feature_batch, test_label_batch, test_name_batch = read_and_decode(val_file_queue,
#                                                                                               flags.batchsize)

# LOG
log_prefix_name = '{}_{}'.format(FLAGS.name, FLAGS.EX_ID)
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
sess.run(init_op)

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord, sess=sess)

# save ( after session )
Saver = tf.train.Saver(max_to_keep=20, keep_checkpoint_every_n_hours=2)

if FLAGS.model_checkpoint_path is not None:
    print('load model from {} ...'.format(FLAGS.model_checkpoint_path))
    Saver.restore(sess=sess, save_path=FLAGS.model_checkpoint_path)
    print('Success !!!')

pylog.info('train_config: {}'.format(FLAGS.YAML))

cnt = 0
for epoch in range(FLAGS.num_epochs + 1):

    # loop = len(train_items) // batchsize
    pylog.info('epoch: {} ... '.format(epoch))

    loop = 2222
    for i in range(loop):
        input_features, input_target_labels, input_video_frames, train_name = sess.run(
            [train_feature_batch, train_label_batch, train_frame_len_batch, train_name_batch])
        input_features, input_target_labels, input_video_frames = split_into_small_peice(input_features,
                                                                                         input_target_labels,
                                                                                         input_video_frames)
        fd = {inputs: input_features, target_labels: input_target_labels, num_frames: input_video_frames}
        loss_value, _ = sess.run([loss, train_op], feed_dict=fd)
        pylog.info('cnt: {} train_loss: {}'.format(cnt, loss_value))
        cnt += 1

coord.request_stop()
coord.join(threads)
