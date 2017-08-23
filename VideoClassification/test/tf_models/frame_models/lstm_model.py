import sys
from VideoClassification.test.tf_models import models
from VideoClassification.test.tf_models import model_utils
import math
import numpy as np
import tensorflow as tf
from VideoClassification.test.tf_models import utils
import tensorflow.contrib.slim as slim

from VideoClassification.test.tf_models.video_models.moe_model import MoeModel

# from tensorflow import flags
# FLAGS = flags.FLAGS

class LstmModel(models.BaseModel):

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
        lstm_size = 32
        number_of_layers = 10

        ## Batch normalize the input
        stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.BasicLSTMCell(
                    lstm_size, forget_bias=1.0, state_is_tuple=False)
                for _ in range(number_of_layers)
            ],
            state_is_tuple=False)

        with tf.variable_scope("RNN"):
            outputs, state = tf.nn.dynamic_rnn(stacked_lstm, model_input,
                                               sequence_length=num_frames,
                                               swap_memory=True,
                                               dtype=tf.float32)

        # return outputs,state

        # from Frame Model to Video Frame to get predict output

        # aggregated_model = getattr(video_level_models,
        #                            FLAGS.video_level_classifier_model)
        aggregated_model = MoeModel
        predictions = aggregated_model().create_model(
            model_input=state,
            original_input=model_input,
            vocab_size=vocab_size,
            **unused_params)
        print(predictions)
        return predictions


if __name__=='__main__':

    x = tf.random_normal(shape=(4,10,1024))
    vocab_size = 101
    num_frames = np.ones(shape=(4,))*10
    num_frames = tf.stack(num_frames)

    pred = LstmModel().create_model(x,vocab_size,num_frames)

    sess = tf.InteractiveSession()
    init = tf.global_variables_initializer()

    sess.run(init)

    p = sess.run(pred)

    sess.run(state)

