import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class MyLinear(nn.Module):
    def __init__(self, inputdim, outputdim):
        super(MyLinear, self).__init__()
        self.weight = Parameter(torch.randn(inputdim, outputdim))
        self.bias = Parameter(torch.randn(outputdim))
        self.reset_parameters()

    def forward(self, x):
        return torch.matmul(x, self.weight) + self.bias
        # return F.linear(x,self.weight,self.bias)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)


nn.Linear
