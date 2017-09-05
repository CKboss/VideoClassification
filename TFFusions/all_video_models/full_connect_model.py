import math
import TFFusions.models as models
import tensorflow as tf
import tensorflow.contrib.slim as slim

from TFFusions.Train.load_yaml_to_FLAG import Get_GlobalFLAG
FLAGS = None


class FullConnectModel(models.BaseModel):
    """Logistic model with L2 regularization."""

    def __init__(self):
        global FLAGS
        FLAGS = Get_GlobalFLAG()

    def create_model(self, model_input, vocab_size, l2_penalty=1e-7, original_input=None, input_relu = False, **unused_params):
        """Creates a logistic model.

        Args:
          model_input: 'batch' x 'num_features' matrix of input features.
          vocab_size: The number of classes in the dataset.

        Returns:
          A dictionary with a tensor containing the probability predictions of the
          model in the 'predictions' key. The dimensions of the tensor are
          batch_size x num_classes."""

        input_relu = FLAGS.input_relu or input_relu
        if input_relu :
            model_input = tf.nn.relu(model_input)

        output = slim.fully_connected( model_input, vocab_size, weights_regularizer=slim.l2_regularizer(l2_penalty))
        return {"predictions": output}
