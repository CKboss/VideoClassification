import math
import TFFusions.models as models
import tensorflow as tf
from tensorflow import flags
import tensorflow.contrib.slim as slim
FLAGS = flags.FLAGS

class ChainSupportReluMoeModel(models.BaseModel):
    """A softmax over a mixture of logistic models (with L2 regularization)."""

    def create_model(self, model_input, vocab_size, num_mixtures=None,
                     l2_penalty=1e-8, sub_scope="", original_input=None, **unused_params):
        num_supports = FLAGS.num_supports
        input_size = model_input.shape.as_list()[1]
        support_relu = slim.fully_connected(
            model_input,
            input_size,
            activation_fn=tf.nn.relu,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="main-relu-"+sub_scope)
        support_predictions = self.sub_model(support_relu, num_supports, sub_scope=sub_scope+"-support")
        main_input = tf.concat([model_input, support_predictions], axis=1)
        main_predictions = self.sub_model(main_input, vocab_size, sub_scope=sub_scope+"-main")
        return {"predictions": main_predictions, "support_predictions": support_predictions}

    def sub_model(self, model_input, vocab_size, num_mixtures=None,
                  l2_penalty=1e-8, sub_scope="", **unused_params):
        num_mixtures = num_mixtures or FLAGS.moe_num_mixtures

        gate_activations = slim.fully_connected(
            model_input,
            vocab_size * (num_mixtures + 1),
            activation_fn=None,
            biases_initializer=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="gates-"+sub_scope)
        expert_activations = slim.fully_connected(
            model_input,
            vocab_size * num_mixtures,
            activation_fn=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="experts-"+sub_scope)

        gating_distribution = tf.nn.softmax(tf.reshape(
            gate_activations,
            [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
        expert_distribution = tf.nn.sigmoid(tf.reshape(
            expert_activations,
            [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures

        final_probabilities_by_class_and_batch = tf.reduce_sum(
            gating_distribution[:, :num_mixtures] * expert_distribution, 1)
        final_probabilities = tf.reshape(final_probabilities_by_class_and_batch,
                                         [-1, vocab_size])
        return final_probabilities


if __name__=='__main__':

    import TFFusions.Config.Config as Config
    from TFFusions.Train.load_yaml_to_FLAG import LOAD_YAML_TO_FLAG,Get_GlobalFLAG

    train_config = Config.TRAIN_SCRIPT + 'chaining-lstm-cnn.yaml'
    LOAD_YAML_TO_FLAG(train_config)
    FLAGS = Get_GlobalFLAG()

    FLAGS.num_supports=3

    inputs = tf.placeholder(dtype=tf.float32,shape=(3,1024))
    num_frames = tf.placeholder(dtype=tf.int32,shape=(3))
    model = ChainSupportReluMoeModel()

    model.create_model(inputs,500)
