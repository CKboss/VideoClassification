import math
import TFFusions.models as models
import tensorflow as tf
import numpy as np
import TFFusions.utils as utils
import tensorflow.contrib.slim as slim

# from tensorflow import flags
# FLAGS = flags.FLAGS

from TFFusions.all_video_models.video_level_models import GetVideoModel
from TFFusions.train_scripts.load_yaml_to_FLAG import Get_GlobalFLAG

FLAGS = None

class LstmCnnDeepCombineChainModel(models.BaseModel):
    """A softmax over a mixture of logistic models (with L2 regularization)."""

    def __init__(self):
        super(LstmCnnDeepCombineChainModel, self).__init__()
        global FLAGS
        FLAGS = Get_GlobalFLAG()

    def cnn(self,
            model_input,
            l2_penalty=1e-8,
            num_filters = [1024, 1024, 1024],
            filter_sizes = [1,2,3],
            sub_scope="",
            **unused_params):
        max_frames = model_input.get_shape().as_list()[1]
        num_features = model_input.get_shape().as_list()[2]

        shift_inputs = []
        for i in range(max(filter_sizes)):
            if i == 0:
                shift_inputs.append(model_input)
            else:
                shift_inputs.append(tf.pad(model_input, paddings=[[0,0],[i,0],[0,0]])[:,:max_frames,:])

        cnn_outputs = []
        for nf, fs in zip(num_filters, filter_sizes):
            sub_input = tf.concat(shift_inputs[:fs], axis=2)
            sub_filter = tf.get_variable(sub_scope+"cnn-filter-len%d"%fs,
                                         shape=[num_features*fs, nf], dtype=tf.float32,
                                         initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1),
                                         regularizer=tf.contrib.layers.l2_regularizer(l2_penalty))
            cnn_outputs.append(tf.einsum("ijk,kl->ijl", sub_input, sub_filter))

        cnn_output = tf.concat(cnn_outputs, axis=2)
        return cnn_output

    def create_model(self, model_input, vocab_size, num_frames, num_mixtures=None,
                     l2_penalty=1e-8, sub_scope="", original_input=None, **unused_params):
        # num_supports = FLAGS.num_supports
        num_layers = FLAGS.deep_chain_layers
        relu_cells = FLAGS.deep_chain_relu_cells
        max_frames = model_input.get_shape().as_list()[1]
        relu_layers = []
        support_predictions = []

        mask = self.get_mask(max_frames, num_frames)
        mean_input = tf.einsum("ijk,ij->ik", model_input, mask) \
                     / tf.expand_dims(tf.cast(num_frames, dtype=tf.float32), dim=1)

        # LSTM
        lstm_output = self.lstmoutput(model_input, vocab_size, num_frames)

        mean_relu = slim.fully_connected(
            mean_input,
            relu_cells,
            activation_fn=tf.nn.relu,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope=sub_scope+"mean-relu")
        mean_relu_norm = tf.nn.l2_normalize(mean_relu, dim=1)
        relu_layers.append(mean_relu_norm)

        cnn_output = self.cnn(lstm_output, num_filters=[relu_cells,2*relu_cells,relu_cells], filter_sizes=[1,2,3], sub_scope=sub_scope+"cnn0")
        max_cnn_output = tf.reduce_max(cnn_output, axis=1)
        normalized_cnn_output = tf.nn.l2_normalize(max_cnn_output, dim=1)
        next_input = normalized_cnn_output

        for layer in range(num_layers):
            sub_prediction = self.sub_model(next_input, vocab_size, sub_scope=sub_scope+"prediction-%d"%layer)
            support_predictions.append(sub_prediction)

            sub_relu = slim.fully_connected(
                sub_prediction,
                relu_cells,
                activation_fn=tf.nn.relu,
                weights_regularizer=slim.l2_regularizer(l2_penalty),
                scope=sub_scope+"relu-%d"%layer)
            relu_norm = tf.nn.l2_normalize(sub_relu, dim=1)
            relu_layers.append(relu_norm)

            cnn_output = self.cnn(lstm_output, num_filters=[relu_cells,relu_cells,relu_cells*2], filter_sizes=[1,2,3], sub_scope=sub_scope+"cnn%d"%(layer+1))
            max_cnn_output = tf.reduce_max(cnn_output, axis=1)
            normalized_cnn_output = tf.nn.l2_normalize(max_cnn_output, dim=1)
            next_input = tf.concat([normalized_cnn_output] + relu_layers, axis=1)

        main_predictions = self.sub_model(next_input, vocab_size, sub_scope=sub_scope+"-main")
        support_predictions = tf.concat(support_predictions, axis=1)
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

    def get_mask(self, max_frames, num_frames):
        mask_array = []
        for i in range(max_frames + 1):
            tmp = [0.0] * max_frames
            for j in range(i):
                tmp[j] = 1.0
            mask_array.append(tmp)
        mask_array = np.array(mask_array)
        mask_init = tf.constant_initializer(mask_array)
        mask_emb = tf.get_variable("mask_emb", shape = [max_frames + 1, max_frames],
                                   dtype = tf.float32, trainable = False, initializer = mask_init)
        mask = tf.nn.embedding_lookup(mask_emb, num_frames)
        return mask


    def lstmoutput(self, model_input, vocab_size, num_frames):

        number_of_layers = FLAGS.lstm_layers

        lstm_sizes = list(map(int, FLAGS.lstm_cells.split(",")))
        feature_names, feature_sizes = utils.GetListOfFeatureNamesAndSizes(
            FLAGS.feature_names, FLAGS.feature_sizes)
        sub_inputs = [tf.nn.l2_normalize(x, dim=2) for x in tf.split(model_input, feature_sizes, axis = 2)]

        assert len(lstm_sizes) == len(feature_sizes), \
            "length of lstm_sizes (={}) != length of feature_sizes (={})".format( \
                len(lstm_sizes), len(feature_sizes))

        outputs = []
        for i in range(len(feature_sizes)):
            with tf.variable_scope("RNN%d" % i):
                sub_input = sub_inputs[i]
                lstm_size = lstm_sizes[i]
                ## Batch normalize the input
                stacked_lstm = tf.contrib.rnn.MultiRNNCell(
                    [
                        tf.contrib.rnn.BasicLSTMCell(
                            lstm_size, forget_bias=1.0, state_is_tuple=True)
                        for _ in range(number_of_layers)
                    ],
                    state_is_tuple=True)

                output, state = tf.nn.dynamic_rnn(stacked_lstm, sub_input,
                                                  sequence_length=num_frames,
                                                  swap_memory=FLAGS.rnn_swap_memory,
                                                  dtype=tf.float32)
                outputs.append(output)

        # concat
        final_output = tf.concat(outputs, axis=2)
        return final_output



if __name__=='__main__':
    '''
    example: rgb+audio
    '''

    import TFFusions.Config.Config as Config
    from TFFusions.train_scripts.load_yaml_to_FLAG import LOAD_YAML_TO_FLAG

    train_config = Config.TRAIN_SCRIPT + 'chaining-lstm-cnn.yaml'
    LOAD_YAML_TO_FLAG(train_config)
    FLAGS = Get_GlobalFLAG()

    inputs = tf.placeholder(dtype=tf.float32,shape=(3,600,1024+128))
    num_frames = tf.placeholder(dtype=tf.int32,shape=(3))
    model = LstmCnnDeepCombineChainModel()

    predict_labels = model.create_model(model_input=inputs, vocab_size=FLAGS.vocab_size, num_frames=num_frames)
    # xxx = model.lstmoutput(inputs,500,num_frames)
