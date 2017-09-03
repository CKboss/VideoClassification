import sys
import TFFusions.models as models
import TFFusions.model_utils as model_utils
import math
import numpy as np
import tensorflow as tf
import TFFusions.utils as utils
import tensorflow.contrib.slim as slim

# from tensorflow import flags
# FLAGS = flags.FLAGS

from TFFusions.all_video_models.video_level_models import GetVideoModel
from TFFusions.Train.load_yaml_to_FLAG import Get_GlobalFLAG

FLAGS = None

class LstmAttentionLstmModel2(models.BaseModel):

    def __init__(self):
        super(LstmAttentionLstmModel2, self).__init__()
        global FLAGS
        FLAGS = Get_GlobalFLAG()

    def create_model(self, model_input, vocab_size, num_frames,noise_level=None,dropout=False,keep_prob=None,**unused_params):
        """Creates a model which uses a stack of LSTMs to represent the video.

        Args:
          model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                       input features.
          vocab_size: The number of classes in the dataset.
          num_frames: A vector of length 'batch' which indicates the number of
               frames for each video (before padding).

        Returns:
          A dictionary with a tensor containing the probability predictions of the
          model in the 'predictions' key. The dimensions of the tensor are
          'batch_size' x 'num_classes'.
        """
        lstm_size = int(FLAGS.lstm_cells)
        number_of_layers = FLAGS.lstm_layers
        l2_penalty = unused_params.get("l2_penalty", 1e-8)
        # max_frames = model_input.get_shape().as_list()[1]
        max_frames = FLAGS.fix_length

        noise_level = noise_level or FLAGS.noise_level
        dropout = dropout or FLAGS.dropout
        if dropout:
            keep_prob = keep_prob or FLAGS.keep_prob

        mask_array = []
        for i in range(max_frames + 1):
            tmp = [0.0] * max_frames
            for j in range(i):
                tmp[j] = 1.0
            mask_array.append(tmp)
        mask_array = np.array(mask_array)
        mask_init = tf.constant_initializer(mask_array)
        mask_emb = tf.get_variable("mask_emb", shape=[max_frames + 1, max_frames],
                                   dtype=tf.float32, trainable=False, initializer=mask_init)

        ## Batch normalize the input

        if dropout:
            stacked_cell = tf.contrib.rnn.MultiRNNCell(
                [
                    tf.contrib.rnn.DropoutWrapper(
                        tf.contrib.rnn.BasicLSTMCell(
                            lstm_size, forget_bias=1.0, state_is_tuple=True),
                        input_keep_prob=keep_prob)
                    for _ in range(number_of_layers)
                ],
                state_is_tuple=True)
        else :
            stacked_cell = tf.contrib.rnn.MultiRNNCell(
                [
                    tf.contrib.rnn.BasicLSTMCell(
                        lstm_size, forget_bias=1.0, state_is_tuple=True)
                    for _ in range(number_of_layers)
                ],
                state_is_tuple=True)

        stacked_att_cell = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.BasicLSTMCell(
                    lstm_size, forget_bias=1.0, state_is_tuple=True)
                for _ in range(number_of_layers)
            ],
            state_is_tuple=True)

        with tf.variable_scope("ATT"):
            att_outputs, att_state = tf.nn.dynamic_rnn(stacked_att_cell, model_input,
                                                       sequence_length=num_frames,
                                                       swap_memory=FLAGS.rnn_swap_memory,
                                                       dtype=tf.float32)
            att_final_memory = tf.concat(list(map(lambda x: x.c, att_state)), axis=1)

        with tf.variable_scope("RNN"):
            outputs, state = tf.nn.dynamic_rnn(stacked_cell, model_input,
                                               sequence_length=num_frames,
                                               swap_memory=FLAGS.rnn_swap_memory,
                                               dtype=tf.float32)
            final_memory = tf.concat(list(map(lambda x: x.c, state)), axis=1)

            mask = tf.expand_dims(tf.nn.embedding_lookup(mask_emb, num_frames), axis=2)
            attention_fc = slim.fully_connected(
                att_outputs, 1, activation_fn=tf.nn.sigmoid,
                weights_regularizer=slim.l2_regularizer(l2_penalty))

            attention = attention_fc * mask
            attention_sum = tf.reduce_sum(attention, axis=1, keep_dims=True) + 1e-8
            attention = attention / attention_sum

            attended_output = tf.reduce_sum(attention * outputs, axis=1)
            final_state = tf.concat([attended_output, att_final_memory, final_memory], axis=1)

        if noise_level is not None:
            final_state = final_state + tf.random_normal(tf.shape(final_state), mean=0.0, stddev=noise_level)

        aggregated_model = GetVideoModel(FLAGS.video_level_model)
        return aggregated_model().create_model(
            model_input=final_state,
            original_input=model_input,
            vocab_size=vocab_size,
            **unused_params)

if __name__=='__main__':

    import TFFusions.Config.Config as Config
    from TFFusions.Train.load_yaml_to_FLAG import LOAD_YAML_TO_FLAG

    train_config = Config.TRAIN_SCRIPT + 'att-lstm.yaml'
    LOAD_YAML_TO_FLAG(train_config)
    FLAGS = Get_GlobalFLAG()

    inputs = tf.placeholder(dtype=tf.float32,shape=(3,600,1024))
    num_frames = tf.placeholder(dtype=tf.int32,shape=(3))
    model = LstmAttentionLstmModel2()

    model.create_model(inputs,500,num_frames)

