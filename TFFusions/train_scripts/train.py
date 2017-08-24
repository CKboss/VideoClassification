import numpy as np
import pprint
import random
import os
import logging

import tensorflow as tf

import TFFusions.Config.Config as Config
from TFFusions.train_scripts.load_yaml_to_FLAG import LOAD_YAML_TO_FLAG,Get_GlobalFLAG
from TFFusions.all_frame_models.frame_level_models import GetFrameModel
from TFFusions.toolkits.dataloader import getTrainItems,getValItems,Load_Features
from TFFusions.losses import SoftmaxLoss
from TFFusions.average_precision_calculator import mean_ap
from TFFusions.Logger import Logger

def find_class_by_name(name,models):
    classes = [getattr(model,name,None) for model in models]
    if len(classes) == 1:
        return classes[0]
    else:
        return classes

def gen_tf_input(items,kind):

    global FLAGS

    features = []
    video_frames = []
    labels = []

    for item in items:
        feat = Load_Features(item[0],kind=kind,limitlen=600)
        video_frames.append(feat.shape[0])
        ax0_len = feat.shape[0]
        feat = np.pad(feat,((0,600-ax0_len),(0,0)),'constant',constant_values=(0,0))
        features.append(feat)
        labels.append(item[1])

    features = np.c_[features]

    target_label = np.zeros((FLAGS.batchsize,500))
    for id,label in enumerate(labels):
        for la in label:
            target_label[id,la] = 1

    return features, video_frames, target_label

def main(config_yaml=None):

    train_config = config_yaml or Config.TRAIN_SCRIPT+'lstm-memory-cell1024.yaml'
    LOAD_YAML_TO_FLAG(train_config)
    FLAGS = Get_GlobalFLAG()

    if os.path.exists(FLAGS.train_dir)==False:
        print('mk train dir {}'.format(FLAGS.train_dir))
        os.mkdir(FLAGS.train_dir)

    Saver = tf.train.Saver(max_to_keep=20,keep_checkpoint_every_n_hours=2)

    train_items = getTrainItems()
    val_items = getValItems()
    batchsize = FLAGS.batchsize

    inputs = tf.placeholder(dtype=tf.float32,shape=(None,600,4096))
    num_frames = tf.placeholder(dtype=tf.int32,shape=(None))
    target_labels = tf.placeholder(dtype=tf.int32,shape=(None,FLAGS.vocab_size))

    model = GetFrameModel(FLAGS.frame_level_model)()
    lossfunc = SoftmaxLoss()

    predict_labels = model.create_model(model_input=inputs, vocab_size=FLAGS.vocab_size, num_frames=num_frames)
    predict_labels = predict_labels['predictions']
    loss = lossfunc.calculate_loss(predict_labels,target_labels)

    optimizer_class = find_class_by_name(FLAGS.optimize,[tf.train])
    train_op = optimizer_class(FLAGS.base_learning_rate).minimize(loss)

    # LOG
    log_prefix_name = '{}_{}'.format(FLAGS.name,FLAGS.EX_ID)

    # python's logging
    pylog = logging.getLogger(log_prefix_name)
    pylog.setLevel(logging.DEBUG)
    fh = logging.FileHandler(FLAGS.train_dir+'/'+log_prefix_name+'.log')
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s-%(name)s-%(levelname)s: %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    pylog.addHandler(fh)
    pylog.addHandler(ch)


    # tfboard's log
    logger = Logger(FLAGS.train_dir+log_prefix_name)

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    sess = tf.Session(config=tf_config)
    sess.run(tf.global_variables_initializer())

    if FLAGS.model_checkpoint_path is not None:
        print('load model from {} ...'.format(FLAGS.model_checkpoint_path))
        Saver.restore(sess=sess,save_path=FLAGS.model_checkpoint_path)
        print('Success !!!')
    else :
        sess.run(tf.global_variables_initializer())

    cnt = 0
    for epoch in range(FLAGS.num_epochs):
        loop = len(train_items)//batchsize
        for i in range(loop):

            l = i*batchsize
            r = l+batchsize
            items = train_items[l:r]
            features, video_frames, target_label = gen_tf_input(items,'train')
            video_frames = np.array(video_frames)

            fd = {inputs:features, target_labels:target_label, num_frames:video_frames}
            loss_value,_ = sess.run([loss,train_op],feed_dict=fd)

            logger.scalar_summary(log_prefix_name+'/train_loss',loss_value,cnt)
            pylog.info('cnt: {} train_loss: {}'.format(cnt,loss_value))

            if cnt%100 == 0:

                fd = {inputs:features, target_labels:target_label, num_frames:video_frames}
                predict = sess.run(predict_labels,feed_dict=fd)
                train_meanap = mean_ap(predict,target_label)
                logger.scalar_summary(log_prefix_name+'/train_mAP',train_meanap,cnt)

                pylog.info('cnt: {} train_mAP: {}'.format(cnt,train_meanap))

                items = random.choices(val_items,k=FLAGS.batchsize)
                features, video_frames, target_label = gen_tf_input(items,'val')

                fd = {inputs:features, target_labels:target_label, num_frames:video_frames}
                predict,test_loss = sess.run([predict_labels,loss],feed_dict=fd)
                test_meanap = mean_ap(predict,target_label)
                logger.scalar_summary(log_prefix_name+'/test_mAP',test_meanap,cnt)
                logger.scalar_summary(log_prefix_name+'/test_loss',test_loss,cnt)

                pylog.info('cnt: {} test_mAP: {}'.format(cnt,test_meanap))

            if cnt%2000 == 0:
                savepath = FLAGS.train_dir+log_prefix_name+'_save{:03}.ckpt'.format(cnt)
                Saver.save(sess,savepath,cnt)
                pylog.info('save model:{} at {}.'.format(FLAGS.name,savepath))

            cnt+=1


if __name__=='__main__':
    main()
