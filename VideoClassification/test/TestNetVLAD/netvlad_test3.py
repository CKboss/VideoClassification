import os

import numpy as np
import tensorflow as tf

VGG_MEAN = [103.939, 116.779, 123.68]


class Netvlad:
    """
    A trainable version NetVlad
    """

    def __init__(self, vgg16_npy_path=None, trainable=True, dropout=0.5):
        if vgg16_npy_path is not None:
            self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()
            print("npy file loaded")
        else:
            self.data_dict = None

        self.var_dict = {}
        self.trainable = trainable
        self.dropout = dropout

    def build(self, rgb, train_mode=None):
        """
        load variable from npy to build the VGG

        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        :param train_mode: a bool tensor, usually a placeholder: if True, dropout will be turned on
        """

        rgb_scaled = rgb * 255.0

        # Convert RGB to BGR
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
        assert red.get_shape().as_list()[1:] == [224, 224, 1]
        assert green.get_shape().as_list()[1:] == [224, 224, 1]
        assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        bgr = tf.concat(axis=3, values=[
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])
        assert bgr.get_shape().as_list()[1:] == [224, 224, 3]

        self.conv1_1 = self.conv_layer(bgr, 3, 64, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, 64, 64, "conv1_2")
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, 64, 128, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, 128, 128, "conv2_2")
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, 128, 256, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, 256, 256, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, 256, 256, "conv3_3")
        self.pool3 = self.max_pool(self.conv3_3, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, 256, 512, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, 512, 512, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, 512, 512, "conv4_3")
        self.pool4 = self.max_pool(self.conv4_3, 'pool4')

        self.conv5_1 = self.conv_layer(self.pool4, 512, 512, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, 512, 512, "conv5_2")
        self.conv5_3 = self.conv_layer_5(self.conv5_2, 512, 512, "conv5_3")

        self.vlad_output = self.vlad_pooling_layer(self.conv5_3, 64, 100, "vlad_pooling")

        # self.vlad_output = self.fc_layer(self.vlad_output_1, 32768, 4096, "vlad_fc")

        self.data_dict = None

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, in_channels, out_channels, name):
        with tf.variable_scope(name):
            filt, conv_biases = self.get_conv_var(3, in_channels, out_channels, name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)

            return relu

    def conv_layer_5(self, bottom, in_channels, out_channels, name):
        with tf.variable_scope(name):
            filt, conv_biases = self.get_conv_var(3, in_channels, out_channels, name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, conv_biases)

            return bias

    def vlad_pooling_layer(self, bottom, k_cluster, alpha, name):
        with tf.variable_scope(name):
            filt, conv_biases, centers = self.get_vald_pooling_var(k_cluster, alpha, name)

            conv_reshape = tf.reshape(bottom,
                                      shape=[-1, (bottom.get_shape().as_list()[1] * bottom.get_shape().as_list()[2]),
                                             512], name='reshape')  # conv_reshape is B x N x D
            conv_norm = tf.nn.l2_normalize(conv_reshape, dim=1)
            descriptor = tf.expand_dims(conv_norm, axis=-1, name='expanddim')  # descriptor is B x N x D x 1
            conv_vlad = tf.nn.convolution(descriptor, filt, padding='VALID')  # conv_vlad is B x N x 1 x K
            bias = tf.nn.bias_add(conv_vlad, conv_biases)
            a_k = tf.nn.softmax(tf.squeeze(bias, axis=2), dim=-1, name="vlad_softmax")  # a_k is B x N x K

            V1 = tf.matmul(conv_reshape, a_k, transpose_a=True)  # V_1 is B x D x K
            V2 = tf.multiply(tf.reduce_sum(a_k, axis=1, keep_dims=True), centers)  # V_1 is B x D x K
            V = tf.subtract(V1, V2)

            norm = tf.nn.l2_normalize(tf.reshape(tf.nn.l2_normalize(V, dim=1), shape=[-1, 32768]),
                                      dim=1)  # norm is B x (D x K)

            return norm

    def fc_layer(self, bottom, in_size, out_size, name):
        with tf.variable_scope(name):
            weights, biases = self.get_fc_var(in_size, out_size, name)

            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_conv_var(self, filter_size, in_channels, out_channels, name):
        initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, 0.001)
        filters = self.get_var(initial_value, name, 0, name + "_filters")

        initial_value = tf.truncated_normal([out_channels], .0, .001)
        biases = self.get_var(initial_value, name, 1, name + "_biases")

        return filters, biases

    def get_vald_pooling_var(self, k_cluster, alpha, name):
        initial_value = tf.truncated_normal([1, 512, 1, k_cluster], 0.0, 0.001)
        filters = self.get_var(initial_value, name, 0, name + "_filters")

        initial_value = tf.truncated_normal([512, k_cluster], 0.0, 0.001)
        centers = self.get_var(initial_value, name, 1, name + '_centers')

        initial_value = tf.truncated_normal([k_cluster], 0.0, 0.001)
        biases = self.get_var(initial_value, name, 2, name + '_biases')

        return filters, biases, centers

    def get_fc_var(self, in_size, out_size, name):
        initial_value = tf.truncated_normal([in_size, out_size], 0.0, 0.1)
        weights = self.get_var(initial_value, name, 0, name + "_weights")

        initial_value = tf.truncated_normal([out_size], .0, .1)
        biases = self.get_var(initial_value, name, 1, name + "_biases")

        return weights, biases

    def get_var(self, initial_value, name, idx, var_name):
        if self.data_dict is not None and name in self.data_dict:
            value = self.data_dict[name][idx]
        else:
            value = initial_value

        if self.trainable:
            var = tf.Variable(value, name=var_name)
        else:
            var = tf.constant(value, dtype=tf.float32, name=var_name)

        self.var_dict[(name, idx)] = var

        # print var_name, var.get_shape().as_list()
        assert var.get_shape() == initial_value.get_shape()

        return var

    def save_npy(self, sess, npy_path="./netvlad-save.npy"):
        assert isinstance(sess, tf.Session)

        data_dict = {}

        for (name, idx), var in list(self.var_dict.items()):
            var_out = sess.run(var)
            if name not in data_dict:
                data_dict[name] = {}
            data_dict[name][idx] = var_out

        np.save(npy_path, data_dict)
        print(("file saved", npy_path))
        return npy_path

    def get_var_count(self):
        count = 0
        for v in list(self.var_dict.values()):
            count += tf.reduce(lambda x, y: x * y, v.get_shape().as_list())
        return count


if __name__ == '__main__':
    x = tf.random_normal(shape=(4, 224, 224, 3))

    net = Netvlad()

    net.build(x)

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    conv5 = sess.run(net.conv5_3)
    vlad_out = sess.run(net.vlad_output)
