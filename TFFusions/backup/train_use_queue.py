import numpy as np
import pprint
import random
import os
import logging

import tensorflow as tf

import TFFusions.Config.Config as Config
from TFFusions.train_scripts.load_yaml_to_FLAG import LOAD_YAML_TO_FLAG, Get_GlobalFLAG
from TFFusions.all_frame_models.frame_level_models import GetFrameModel
from TFFusions.toolkits.dataloader import getTrainItems, getValItems, gen_tf_input, PictureQueue
from TFFusions.losses import SoftmaxLoss
from TFFusions.average_precision_calculator import mean_ap, accuracy
from TFFusions.Logger import Logger


def find_class_by_name(name, models):
    classes = [getattr(model, name, None) for model in models]
    if len(classes) == 1:
        return classes[0]
    else:
        return classes


def main(config_yaml=None):
    train_config = config_yaml or Config.TRAIN_SCRIPT + 'lstm-memory-cell1024.yaml'
    LOAD_YAML_TO_FLAG(train_config)
    FLAGS = Get_GlobalFLAG()

    if os.path.exists(FLAGS.train_dir) == False:
        print('mk train dir {}'.format(FLAGS.train_dir))
        os.mkdir(FLAGS.train_dir)

    train_items = getTrainItems()
    val_items = getValItems()
    batchsize = FLAGS.batchsize

    inputs = tf.placeholder(dtype=tf.float32, shape=(None, 600, 4096))
    num_frames = tf.placeholder(dtype=tf.int32, shape=(None))
    target_labels = tf.placeholder(dtype=tf.int32, shape=(None, FLAGS.vocab_size))

    model = GetFrameModel(FLAGS.frame_level_model)()
    lossfunc = SoftmaxLoss()

    predict_labels = model.create_model(model_input=inputs, vocab_size=FLAGS.vocab_size, num_frames=num_frames)
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
    sess = tf.Session(config=tf_config)
    sess.run(tf.global_variables_initializer())

    # Load Queue
    pq_train = PictureQueue(kind='train', batchsize=batchsize, worker=5)
    pq_test = PictureQueue(kind='val', batchsize=batchsize, worker=3)

    # save ( after session )
    Saver = tf.train.Saver(max_to_keep=20, keep_checkpoint_every_n_hours=2)

    if FLAGS.model_checkpoint_path is not None:
        print('load model from {} ...'.format(FLAGS.model_checkpoint_path))
        Saver.restore(sess=sess, save_path=FLAGS.model_checkpoint_path)
        print('Success !!!')

    cnt = 0

    for epoch in range(FLAGS.num_epochs):
        loop = len(train_items) // batchsize
        for i in range(loop):

            # l = i*batchsize
            # r = l+batchsize
            # items = train_items[l:r]
            # features, video_frames, target_label = gen_tf_input(items,'train')

            features, video_frames, target_label = pq_train.Get()

            video_frames = np.array(video_frames)

            fd = {inputs: features, target_labels: target_label, num_frames: video_frames}
            loss_value, _ = sess.run([loss, train_op], feed_dict=fd)

            logger.scalar_summary(log_prefix_name + '/train_loss', loss_value, cnt)
            pylog.info('cnt: {} train_loss: {}'.format(cnt, loss_value))

            if cnt % 50 == 0:
                fd = {inputs: features, target_labels: target_label, num_frames: video_frames}
                predict = sess.run(predict_labels, feed_dict=fd)
                train_meanap = mean_ap(predict, target_label)
                acc = accuracy(predict, target_labels, topk=(1, 5, 10))

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
                features, video_frames, target_label = pq_test.Get()

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


if __name__ == '__main__':
    main()
