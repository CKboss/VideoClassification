import numpy as np
import pprint
import random
import os
import logging
import glob

import tensorflow as tf

import TFFusions.Config.Config as Config
from TFFusions.train_scripts.load_yaml_to_FLAG import LOAD_YAML_TO_FLAG, Get_GlobalFLAG
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

def split_into_small_peice(features,target_label,video_frames,fix_lenght=10,scale=8):
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
    try:
        fix_lenght = FLAGS.fix_length
        scale = FLAGS.scale
    except:
        fix_lenght = 10
        scale = 8

    n = features.shape[0]
    n2 = target_label.shape[-1]
    m = features.shape[-1]

    features_ret = []
    target_label_ret = []
    video_frames_ret = []

    for i in range(n):

        video_len = video_frames[i]
        rid = random.choices(list(range(video_len)),k=scale)

        # print('video_len: ',video_len)
        if video_len <= fix_lenght:
            video_len = fix_lenght+1

        for rg in rid:

            l = rg
            r = rg+fix_lenght-1

            if r >= video_len :
                l = video_len - fix_lenght - 1
                r = video_len - 2

            features_ret.append( features[i,l:r+1,:] )
            video_frames_ret.append(fix_lenght)
            target_label_ret.append(target_label[i])

    features_ret = np.vstack(features_ret).reshape(-1,fix_lenght,m)
    target_label_ret = np.vstack(target_label_ret).reshape(-1,n2)
    video_frames_ret = np.array(video_frames_ret)

    return features_ret,target_label_ret,video_frames_ret

#############################################################################


def main(config_yaml=None):

    global FLAGS

    train_config = config_yaml or Config.TRAIN_SCRIPT + 'lstm-memory-cell1024.yaml'
    LOAD_YAML_TO_FLAG(train_config)
    FLAGS = Get_GlobalFLAG()

    if os.path.exists(FLAGS.train_dir) == False:
        print('mk train dir {}'.format(FLAGS.train_dir))
        os.mkdir(FLAGS.train_dir)

    # train_items = getTrainItems()
    # val_items = getValItems()
    batchsize = FLAGS.batchsize

    #
    if FLAGS.device_id != None:
        os.environ['CUDA_VISIBLE_DEVICES']=str(FLAGS.device_id)[1:-1]

    inputs = tf.placeholder(dtype=tf.float32, shape=(batchsize*FLAGS.scale, FLAGS.fix_length, 4096))
    num_frames = tf.placeholder(dtype=tf.int32, shape=(batchsize*FLAGS.scale))
    target_labels = tf.placeholder(dtype=tf.int32, shape=(batchsize*FLAGS.scale, FLAGS.vocab_size))

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
    val_file_list = glob.glob('/mnt/md0/LSVC/tfrecords/val_*')
    train_file_list = glob.glob('/mnt/md0/LSVC/tfrecords/train_*')
    train_file_queue = tf.train.string_input_producer(train_file_list)
    val_file_queue = tf.train.string_input_producer(val_file_list)
    train_frame_len_batch, train_feature_batch, train_label_batch, train_name_batch = read_and_decode(train_file_queue,
                                                                                                      FLAGS.batchsize)
    test_frame_len_batch, test_feature_batch, test_label_batch, test_name_batch = read_and_decode(val_file_queue,
                                                                                                  FLAGS.batchsize)

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

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    # save ( after session )
    Saver = tf.train.Saver(max_to_keep=20, keep_checkpoint_every_n_hours=2)

    if FLAGS.model_checkpoint_path is not None:
        print('load model from {} ...'.format(FLAGS.model_checkpoint_path))
        Saver.restore(sess=sess, save_path=FLAGS.model_checkpoint_path)
        print('Success !!!')

    cnt = 0

    for epoch in range(FLAGS.num_epochs+1):

        # loop = len(train_items) // batchsize
        loop = 2222
        pylog.info('epoch: {} ... '.format(epoch))

        for i in range(loop):

            features, target_label, video_frames, train_name = sess.run( [train_feature_batch, train_label_batch, train_frame_len_batch, train_name_batch])
            # F = features
            # T = target_label
            # V = video_frames
            # print(features.shape)
            features, target_label, video_frames = split_into_small_peice(features,target_label,video_frames)
            # print(features.shape)

            fd = {inputs: features, target_labels: target_label, num_frames: video_frames}

            loss_value, _ = sess.run([loss, train_op], feed_dict=fd)

            logger.scalar_summary(log_prefix_name + '/train_loss', loss_value, cnt)
            pylog.info('cnt: {} train_loss: {}'.format(cnt, loss_value))

            if cnt % 30 == 0:
                features, target_label, video_frames, train_name = sess.run(
                    [train_feature_batch, train_label_batch, train_frame_len_batch, train_name_batch])

                features, target_label, video_frames = split_into_small_peice(features,target_label,video_frames)
                fd = {inputs: features, target_labels: target_label, num_frames: video_frames}

                predict = sess.run(predict_labels, feed_dict=fd)
                train_meanap = mean_ap(predict, target_label)
                acc = accuracy(predict, target_label, topk=(1, 5, 10))

                logger.scalar_summary(log_prefix_name + '/train_mAP', train_meanap, cnt)
                logger.scalar_summary(log_prefix_name + '/train_acc@1', acc[0], cnt)
                logger.scalar_summary(log_prefix_name + '/train_acc@5', acc[1], cnt)
                logger.scalar_summary(log_prefix_name + '/train_acc@10', acc[2], cnt)

                pylog.info('cnt: {} train_mAP: {}'.format(cnt, train_meanap))
                pylog.info('cnt: {} train_acc@1: {}'.format(cnt, acc[0]))
                pylog.info('cnt: {} train_acc@5: {}'.format(cnt, acc[1]))
                pylog.info('cnt: {} train_acc@10: {}'.format(cnt, acc[2]))

                # items = random.choices(val_items,k=FLAGS.batchsize)
                # features, video_frames, target_label = gen_tf_input(items,'val')
                # features, video_frames, target_label = pq_test.Get()

                features, target_label, video_frames, test_name = sess.run(
                    [test_feature_batch, test_label_batch, test_frame_len_batch, test_name_batch])

                features, target_label, video_frames = split_into_small_peice(features,target_label,video_frames)

                fd = {inputs: features, target_labels: target_label, num_frames: video_frames}
                predict, test_loss = sess.run([predict_labels, loss], feed_dict=fd)
                test_meanap = mean_ap(predict, target_label)
                acc = accuracy(predict, target_label, topk=(1, 5, 10))

                logger.scalar_summary(log_prefix_name + '/test_mAP', test_meanap, cnt)
                logger.scalar_summary(log_prefix_name + '/test_acc@1', acc[0], cnt)
                logger.scalar_summary(log_prefix_name + '/test_acc@5', acc[1], cnt)
                logger.scalar_summary(log_prefix_name + '/test_acc@10', acc[2], cnt)
                logger.scalar_summary(log_prefix_name + '/test_loss', test_loss, cnt)

                pylog.info('cnt: {} test_mAP: {}'.format(cnt, test_meanap))
                pylog.info('cnt: {} test_loss: {}'.format(cnt, test_loss))
                pylog.info('cnt: {} test_acc@1: {}'.format(cnt, acc[0]))
                pylog.info('cnt: {} test_acc@5: {}'.format(cnt, acc[1]))
                pylog.info('cnt: {} test_acc@10: {}'.format(cnt, acc[2]))

            if cnt % 2000 == 0:
                savepath = FLAGS.train_dir + log_prefix_name + '_save{:03}.ckpt'.format(cnt)
                Saver.save(sess, savepath, cnt)
                pylog.info('save model:{} at {}.'.format(FLAGS.name, savepath))

            cnt += 1

    coord.request_stop()
    coord.join(threads)


if __name__ == '__main__':
    main()
