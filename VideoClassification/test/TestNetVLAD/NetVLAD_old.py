import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.functional as F
from torch.autograd import Variable

import numpy as np


def L2_Normalize(x: torch.Tensor):
    if not (isinstance(x, torch.DoubleTensor) or isinstance(x, torch.FloatTensor)):
        x = x.float()

    if len(x.size()) == 1:
        x = x.view(1, -1)
    norm = torch.sqrt(torch.sum(x * x, dim=1))
    norm = norm.view(-1, 1)
    return x / norm


def L1_Normalize(x: torch.Tensor):
    if not (isinstance(x, torch.DoubleTensor) or isinstance(x, torch.FloatTensor)):
        x = x.float()

    if len(x.size()) == 1:
        x = x.view(1, -1)
    norm = torch.sum(torch.abs(x), dim=1)
    norm = norm.view(-1, 1)
    return x / norm


class PoolingBaseModel(nn.Module):
    def __init__(self, feature_size, max_samples, cluster_size, output_dim, gating=True, add_batch_norm=True):

        super(PoolingBaseModel, self).__init__()

        self.feature_size = feature_size
        self.max_sample = max_samples
        self.cluster_size = cluster_size
        self.ouput_dim = output_dim
        self.gating = gating
        self.add_batch_norm = add_batch_norm

        self.gating_weights = None
        self.gating_biases = None
        self.bn1 = None

    def forward(self, reshaped_input):
        raise NotImplementedError

    def context_gating(self, input_layer):
        '''
        :param input_layer: batch_size x number_of_activation
        :return:  batch_size = number_of_activation
        '''

        b, n = input_layer.size()

        if self.gating_weights is None:
            self.gating_weights = Parameter(torch.Tensor(n, n))
            stdv = 1. / math.sqrt(self.gating_weights.size(1))
            self.gating_weights.data.uniform_(-stdv, stdv)

        gates = F.matmul(input_layer, self.gating_weights)

        if self.add_batch_norm:
            if self.bn1 is None:
                self.bn1 = nn.BatchNorm1d(n)
            gates = self.bn1(gates)
        else:
            if self.gating_biases is None:
                self.gating_biases = Parameter(torch.Tensor(n))
                stdv = 1 / math.sqrt(n)
                self.gating_biases.data.uniform_(-stdv, stdv)
            gates += self.gating_biases

        activation = F.mul(input_layer, gates)

        return activation


class NetVLAD(PoolingBaseModel):
    """NetVLAD
    """

    def __init__(self, feature_size, max_samples, cluster_size, output_dim, gating=True, add_batch_norm=True):

        super(NetVLAD, self).__init__(
            feature_size=feature_size,
            max_samples=max_samples,
            cluster_size=cluster_size,
            output_dim=output_dim,
            gating=True,
            add_batch_norm=True)

        self.cluster_weight_1 = None
        self.cluster_biase_1 = None
        self.bn1 = None
        self.cluster_weight_2 = None

        self.softmax = nn.Softmax()

    def forward(self, reshaped_input):

        '''
        :param reshaped_input: batch_size x max_sample x features_size
        :return: batch_size x output_dim
        '''

        if self.cluster_weight_1 is None:
            self.cluster_weight_1 = Parameter(torch.Tensor(self.feature_size, self.cluster_size))
            stdv = 1. / math.sqrt(self.cluster_size)
            self.cluster_weight_1.data.uniform_(-stdv, stdv)

        print('reshaped_input:', reshaped_input.size())
        print('cluster_weight_1:', self.cluster_weight_1.size())

        activation = F.matmul(reshaped_input, self.cluster_weight_1)

        print('activation:', activation.size())

        if self.add_batch_norm:
            if self.bn1 is None:
                self.bn1 = nn.BatchNorm1d(self.cluster_size)
            activation = self.bn1(activation)
        else:
            if self.cluster_biase_1 is None:
                self.cluster_biase_1 = Parameter(torch.Tensor(self.cluster_size))
            activation = activation + self.cluster_biase_1

        activation = self.softmax(activation)
        activation = activation.view(-1, self.max_sample, self.cluster_size)

        a_sum = torch.sum(activation, -2)

        if self.cluster_weight_2 is None:
            self.cluster_weight_2 = Parameter(torch.Tensor(self.feature_size, self.cluster_size))
            stdv = 1. / math.sqrt(self.cluster_size)
            self.cluster_weight_2.data.uniform_(-stdv, stdv)

        a = torch.mul(a_sum, self.cluster_weight_2)

        activation = torch.transpose(activation, 2, 1)

        vlad = torch.matmul(activation, reshaped_input)
        print('vlad:', vlad.size())
        vlad = torch.transpose(vlad, 2, 1)
        vlad = vlad - a

        vlad = L2_Normalize(vlad[0])

        return vlad


if __name__ == '__main__':
    x = torch.randn(10, 1024)
    x = Variable(x)

    x2 = torch.randn(13, 1024)
    x2 = Variable(x2)

    base = PoolingBaseModel(feature_size=1024, max_samples=10, cluster_size=64, output_dim=10, gating=True)

    netvlad = NetVLAD(feature_size=1024, max_samples=10, cluster_size=64, output_dim=10, gating=True)

    y = netvlad.forward(x)

    print(y.size())
