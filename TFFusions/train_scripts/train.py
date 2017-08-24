import numpy as np
import pprint
import os

import tensorflow as tf

import TFFusions.Config.Config as Config
from TFFusions.train_scripts.load_yaml_to_FLAG import LOAD_YAML_TO_FLAG,Get_GlobalFLAG
from TFFusions.all_frame_models.frame_level_models import GetFrameModel
from TFFusions.toolkits.dataloader import getTrainItems,getValItems,Load_Features
from TFFusions.losses import SoftmaxLoss

def find_class_by_name(name,models):
    classes = [getattr(model,name,None) for model in models]
    return classes

def main():

    train_config = Config.TRAIN_SCRIPT+'lstm-memory-cell1024.yaml'
    LOAD_YAML_TO_FLAG(train_config)
    FLAGS = Get_GlobalFLAG()

    if os.path.exists(FLAGS.train_dir)==False:
        print('mk train dir {}'.format(FLAGS.train_dir))
        os.mkdir(FLAGS.train_dir)

    train_items = getTrainItems()
    val_items = getValItems()
    batchsize = FLAGS.batchsize

    inputs = tf.placeholder(dtype=tf.float32,shape=(None,600,4096))
    num_frames = tf.placeholder(dtype=tf.int32,shape=(None))
    target_labels = tf.placeholder(dtype=tf.int32,shape=(None,501))

    model = GetFrameModel(FLAGS.frame_level_model)()
    lossfunc = SoftmaxLoss()

    predict_labels = model.create_model(model_input=inputs, vocab_size=501, num_frames=num_frames)
    train_loss = lossfunc.calculate_loss(predict_labels,target_labels)

    optimizer_class = find_class_by_name(FLAGS.optimize,[tf.train])


    for epoch in range(FLAGS.num_epochs):

        loop = len(train_items)//batchsize

        for i in range(loop):

            l = i*batchsize
            r = l+batchsize
            items = train_items[l:r]

            features = []
            labels = []
            video_frames = []

            for item in items:
                feat = Load_Features(item[0],kind='train',limitlen=600)

                video_frames.append(feat.shape[0])
                ax0_len = feat.shape[0]
                feat = np.pad(feat,((0,600-ax0_len),(0,0)),'constant',constant_values=(0,0))
                features.append(feat)
                labels.append(item[1][0])

            features = np.c_[features]
            labels = np.array(labels)
            video_frames = np.array(video_frames)

            break
        break


if __name__=='__main__':
    main()
