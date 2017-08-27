import sys
import TFFusions.models as models
import TFFusions.model_utils as model_utils
import math
import numpy as np
import TFFusions.video_level_models
import tensorflow as tf
import TFFusions.utils as utils
import tensorflow.contrib.slim as slim

# from tensorflow import flags
# FLAGS = flags.FLAGS

from TFFusions.all_video_models.video_level_models import GetVideoModel
from TFFusions.train_scripts.load_yaml_to_FLAG import Get_GlobalFLAG
FLAGS = None

class GruPoolingModel(models.BaseModel):

    def __init__(self):
        super(GruPoolingModel,self).__init__()
        global FLAGS
        FLAGS =Get_GlobalFLAG()

    def create_model(self, model_input, vocab_size, num_frames, **unused_params):
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
        gru_size = FLAGS.gru_cells
        number_of_layers = FLAGS.gru_layers

        ## Batch normalize the input
        stacked_gru = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.GRUCell(gru_size)
                for _ in range(number_of_layers)
            ],
            state_is_tuple=False)

        # loss = 0.0
        with tf.variable_scope("RNN"):
            outputs, state = tf.nn.dynamic_rnn(stacked_gru, model_input,
                                               sequence_length=num_frames,
                                               time_major=False,
                                               swap_memory=FLAGS.rnn_swap_memory,
                                               dtype=tf.float32)
            num_frames_matrix = tf.maximum(tf.cast(tf.expand_dims(num_frames, axis=1), dtype=tf.float32), tf.ones([batch_size, 1]))
            pooling_output = tf.reduce_sum(outputs, axis = 1) / num_frames_matrix

        # aggregated_model = getattr(video_level_models,
        #                            FLAGS.video_level_classifier_model)

        aggregated_model = GetVideoModel(FLAGS.video_level_model)
        return aggregated_model().create_model(
            model_input=pooling_output,
            original_input=model_input,
            vocab_size=vocab_size,
            **unused_params)

