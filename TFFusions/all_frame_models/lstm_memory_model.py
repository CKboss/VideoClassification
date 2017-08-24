import sys
import TFFusions.models as models
import TFFusions.model_utils
import math
import numpy as np
import tensorflow as tf
import TFFusions.utils as units
import tensorflow.contrib.slim as slim

# from tensorflow import flags
# FLAGS = flags.FLAGS

from TFFusions.all_video_models.video_level_models import GetVideoModel
from TFFusions.train_scripts.load_yaml_to_FLAG import Get_GlobalFLAG

FLAGS = None

class LstmMemoryModel(models.BaseModel):

    def __init__(self):
        super(LstmMemoryModel,self).__init__()
        global FLAGS
        FLAGS = Get_GlobalFLAG()

    def create_model(self, model_input, vocab_size, num_frames,
                     dropout=False, keep_prob=None, noise_level=None,
                     **unused_params):
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

        ## Batch normalize the input
        if dropout:
            stacked_lstm = tf.contrib.rnn.MultiRNNCell(
                [
                    tf.contrib.rnn.DropoutWrapper(
                        tf.contrib.rnn.BasicLSTMCell(
                            lstm_size, forget_bias=1.0, state_is_tuple=True),
                        input_keep_prob=keep_prob)
                    for _ in range(number_of_layers)
                ],
                state_is_tuple=True)
        else:
            stacked_lstm = tf.contrib.rnn.MultiRNNCell(
                [
                    tf.contrib.rnn.BasicLSTMCell(
                        lstm_size, forget_bias=1.0, state_is_tuple=True)
                    for _ in range(number_of_layers)
                ],
                state_is_tuple=True)

        with tf.variable_scope("RNN"):
            outputs, state = tf.nn.dynamic_rnn(stacked_lstm, model_input,
                                               sequence_length=num_frames,
                                               swap_memory=FLAGS.rnn_swap_memory,
                                               dtype=tf.float32)
            final_state = tf.concat(list(map(lambda x: x.c, state)), axis = 1)

        if noise_level is not None:
            final_state = final_state + tf.random_normal(tf.shape(final_state), mean=0.0, stddev=noise_level)

        # aggregated_model = getattr(video_level_models, FLAGS.video_level_classifier_model)

        aggregated_model = GetVideoModel(FLAGS.video_level_model)

        return aggregated_model().create_model(
            model_input=final_state,
            original_input=model_input,
            vocab_size=vocab_size,
            num_frames=num_frames,
            **unused_params)

