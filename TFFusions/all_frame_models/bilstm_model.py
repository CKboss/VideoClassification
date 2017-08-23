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
from TFFusions.train_scripts.load_yaml_to_FLAG import Get_GlobalFLAG

FLAGS = None

class BiLstmModel(models.BaseModel):

    def __init__(self):
        super(BiLstmModel,self).__init__()
        FLAGS = Get_GlobalFLAG()

    def create_model(self, model_input, vocab_size, num_frames, **unused_params):
        """Creates a model which uses a stack of Bi LSTMs to represent the video.

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
        fw_stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.BasicLSTMCell(
                    lstm_size, forget_bias=1.0, state_is_tuple=False)
                for _ in range(number_of_layers)
            ],
            state_is_tuple=False)
        bw_stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.BasicLSTMCell(
                    lstm_size, forget_bias=1.0, state_is_tuple=False)
                for _ in range(number_of_layers)
            ],
            state_is_tuple=False)

        with tf.variable_scope("RNN"):
            outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw = fw_stacked_lstm, cell_bw = bw_stacked_lstm,
                                                              inputs = model_input,
                                                              sequence_length=num_frames,
                                                              swap_memory=FLAGS.rnn_swap_memory,
                                                              dtype=tf.float32)
        state_fw, state_bw = states
        state = tf.concat([state_fw, state_bw], axis = 1)

        # aggregated_model = getattr(video_level_models,
        #                            FLAGS.video_level_classifier_model)

        aggregated_model = GetVideoModel(FLAGS.video_level_model)

        return aggregated_model().create_model(
            model_input=state,
            original_input=model_input,
            vocab_size=vocab_size,
            **unused_params)

