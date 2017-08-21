from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

from VideoClassification.model.Layers.NetVLAD import NetVLAD
from VideoClassification.test.TestNetVLAD.test_layer import MyLinear

class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc2 = nn.Linear(60, 10)
        self.vlad = NetVLAD(D=20,K=3)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = self.conv2_drop(self.conv2(x))
        x = F.relu(self.vlad(x))
        x = F.dropout(x, training=self.training)
        # print(x.size())
        x = self.fc2(x)
        return F.log_softmax(x)

class Net3(nn.Module):
    def __init__(self):
        super(Net3, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = MyLinear(320, 50)
        self.fc2 = MyLinear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

class Net_fc(nn.Module):

    def __init__(self):
        super(Net_fc,self).__init__()
        self.fc1 = nn.Linear(784,400)
        self.fc2 = nn.Linear(400,100)
        self.fc3 = nn.Linear(100,10)

    def forward(self,x):
        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.log_softmax(x)
        return x

class Net_fc2(nn.Module):

    def __init__(self):
        super(Net_fc2,self).__init__()
        self.fc1 = MyLinear(784,400)
        self.fc2 = MyLinear(400,100)
        self.fc3 = MyLinear(100,10)

    def forward(self,x):
        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.log_softmax(x)
        return x

if __name__=='__main__':
    x = Variable(torch.randn(4,1,28,28))
    net = NetVLAD(D=1,K=3)
    model = Net2()
    y = net(x)
    y = model(x)
    print(y.size())

