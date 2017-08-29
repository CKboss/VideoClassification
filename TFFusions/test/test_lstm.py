'''
Test LSTM
'''
import os
import logging
import glob
import random
import numpy as np

import tensorflow as tf

import TFFusions.Config.Config as Config
from TFFusions.train_scripts.load_yaml_to_FLAG import LOAD_YAML_TO_FLAG, Get_GlobalFLAG
from TFFusions.all_frame_models.frame_level_models import GetFrameModel
from TFFusions.losses import SoftmaxLoss
from TFFusions.average_precision_calculator import mean_ap, accuracy
from TFFusions.Logger import Logger
from TFFusions.tfrecord_tools import read_and_decode

def find_class_by_name(name, models):
    classes = [getattr(model, name, None) for model in models]
    if len(classes) == 1:
        return classes[0]
    else:
        return classes

train_config ='/datacenter/1/LSVC/Code/VideoClassification/TFFusions/train_scripts/train_config_yaml/lstm-memory-cell1024_testuse.yaml'
LOAD_YAML_TO_FLAG(train_config)
FLAGS = Get_GlobalFLAG()

if os.path.exists(FLAGS.train_dir) == False:
    print('mk train dir {}'.format(FLAGS.train_dir))
    os.mkdir(FLAGS.train_dir)

batchsize = FLAGS.batchsize

#
if FLAGS.device_id != None:
    os.environ['CUDA_VISIBLE_DEVICES']=str(FLAGS.device_id)[1:-1]

inputs = tf.placeholder(dtype=tf.float32, shape=(batchsize, 600, 4096))
num_frames = tf.placeholder(dtype=tf.int32, shape=(batchsize))
target_labels = tf.placeholder(dtype=tf.int32, shape=(batchsize, FLAGS.vocab_size))

model = GetFrameModel(FLAGS.frame_level_model)()
lossfunc = SoftmaxLoss()

predict_labels = model.create_model(model_input=inputs, vocab_size=FLAGS.vocab_size, num_frames=num_frames, num_mixtures=FLAGS.moe_num_mixtures)
predict_labels = predict_labels['predictions']
loss = lossfunc.calculate_loss(predict_labels, target_labels)

global_step = tf.Variable(0, trainable=False)
decayed_learning_rate = tf.train.exponential_decay(FLAGS.base_learning_rate,
                                                   global_step,
                                                   FLAGS.decay_at_epoch,
                                                   FLAGS.learning_rate_decay,
                                                   staircase=True)

optimizer_class = find_class_by_name(FLAGS.optimize, [tf.train])
train_op = optimizer_class(decayed_learning_rate).minimize(loss)

# init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
init_op = tf.global_variables_initializer()

# Load from TFRecord
# val_file_list = glob.glob('/mnt/md0/LSVC/tfrecords/val_*')
# train_file_list = glob.glob('/mnt/md0/LSVC/tfrecords/train_*')

# val_file_list = glob.glob('/datacenter/1/LSVC/tfrecords/val_*')
train_file_list = glob.glob('/datacenter/1/LSVC/tfrecords/train_*')

train_file_queue = tf.train.string_input_producer(train_file_list)
# val_file_queue = tf.train.string_input_producer(val_file_list)
train_frame_len_batch, train_feature_batch, train_label_batch, train_name_batch = read_and_decode(train_file_queue, batchsize)
# test_frame_len_batch, test_feature_batch, test_label_batch, test_name_batch = read_and_decode(val_file_queue, batchsize)

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
# tfboard's log
logger = Logger(FLAGS.train_dir + log_prefix_name)

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf_config.allow_soft_placement = True
tf_config.log_device_placement = True

# init session
sess = tf.Session(config=tf_config)
sess.run(init_op)

Saver = tf.train.Saver(keep_checkpoint_every_n_hours=2)

# FLAGS.model_checkpoint_path = '/tmp/test/lstm-memory-cell1024_EX5_save400.ckpt-400'
if FLAGS.model_checkpoint_path is not None:
    print('load model from {} ...'.format(FLAGS.model_checkpoint_path))
    Saver.restore(sess=sess, save_path=FLAGS.model_checkpoint_path)
    print('Success !!!')

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord, sess=sess)
cnt = 0

dataloop = 20
fd = []
for i in range(dataloop):
    features, target_label, video_frames, train_name = sess.run(
        [train_feature_batch, train_label_batch, train_frame_len_batch, train_name_batch])
    tmp = {inputs: features, target_labels: target_label, num_frames: video_frames}
    fd.append(tmp)

def split_into_small_peice(features,target_label,video_frames,fix_lenght=10,scale=8):

    n = features.shape[0]
    n2 = target_label.shape[-1]
    m = features.shape[-1]

    features_ret = []
    target_label_ret = []
    video_frames_ret = []

    for i in range(n):

        video_len = video_frames[i]
        rid = random.choices(list(range(video_len)),k=scale)

        for rg in rid:

            l = rg
            r = rg+fix_lenght-1

            if r >= video_len :
                l = video_len - fix_lenght - 1
                r = video_len - 2

            # assert r-l+1 == 10 , 'not equal 10 {} {} {}'.format(l,r,video_len)
            features_ret.append( features[i,l:r+1,:] )
            video_frames_ret.append(fix_lenght)
            target_label_ret.append(target_label[i])

    features_ret = np.vstack(features_ret).reshape(-1,fix_lenght,m)
    target_label_ret = np.vstack(target_label_ret).reshape(-1,n2)
    video_frames_ret = np.array(video_frames_ret)

    return features_ret,target_label_ret,video_frames_ret

a,b,c = split_into_small_peice(features,target_label,video_frames)

for epoch in range(FLAGS.num_epochs+1):
    loop = 2222
    pylog.info('epoch: {} ... '.format(epoch))
    for i in range(loop):
        loss_value, _, pred = sess.run([loss, train_op, predict_labels], feed_dict=fd[cnt%dataloop])
        logger.scalar_summary(log_prefix_name + '/train_loss', loss_value, cnt)
        mAP = mean_ap(pred,target_label)
        pylog.info('cnt: {} train_loss: {}'.format(cnt, loss_value))
        pylog.info('cnt: {} train_mAP: {}'.format(cnt,mAP))
        if cnt%100 == 0:
            savepath = FLAGS.train_dir + log_prefix_name + '_save{:03}.ckpt'.format(cnt)
            Saver.save(sess, savepath, cnt)
            pylog.info('save model:{} at {}.'.format(FLAGS.name, savepath))
        cnt += 1

coord.request_stop()
coord.join(threads)

