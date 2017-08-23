import numpy as np
import pprint

import tensorflow as tf

import TFFusions.Config.Config as Config
from TFFusions.train_scripts.load_yaml_to_FLAG import LOAD_YAML_TO_FLAG,Get_GlobalFLAG
from TFFusions.all_frame_models.frame_level_models import GetFrameModel

def main():

    train_config = Config.TRAIN_SCRIPT+'lstm-memory-cell1024.yaml'
    LOAD_YAML_TO_FLAG(train_config)
    FLAGS = Get_GlobalFLAG()

    model = GetFrameModel(FLAGS.frame_level_model)()

    x = tf.random_normal(shape=(4,10,4096))
    vocab_size = 101
    num_frames = np.ones(shape=(4,))*10
    num_frames = tf.stack(num_frames)

    x = tf.placeholder(dtype=tf.float32,shape=(4,10,4096))
    num_frames = tf.placeholder(dtype=tf.int32,shape=(4,))

    y = model.create_model(model_input=x,vocab_size=vocab_size,num_frames=num_frames)


if __name__=='__main__':
    main()
