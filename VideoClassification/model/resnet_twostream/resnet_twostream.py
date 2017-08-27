import types

import torch
import torch.nn as nn
from torch.autograd import Variable

import VideoClassification.Config.Config as Config
from VideoClassification.model.resnet_twostream.resnet import resnet101, resnet152, resnet50
from VideoClassification.utils.Others.toolkits import try_to_load_state_dict


class resnet_TwoStreamNet(nn.Module):
    def __init__(self, in_channels=3, dropout=0.9, pretrained=False, level=101):

        super(resnet_TwoStreamNet, self).__init__()

        savefile = None
        if level == 152:
            resnet = resnet152
            savefile = Config.resnet152_pretrainfile
        elif level == 101:
            resnet = resnet101
            savefile = Config.resnet101_pretrainfile
        elif level == 50:
            resnet = resnet50
            savefile = None
        else:
            raise 'level showld be 101 or 152'

        self.resnet = resnet(in_channels=in_channels, num_classes=101)

        if pretrained:
            init = torch.load(savefile)
            if in_channels > 3:
                del init['conv1.weight']
                del init['bn1.running_mean']
                del init['bn1.running_var']
                del init['bn1.weight']
                del init['bn1.bias']

            del init['fc.weight']
            del init['fc.bias']

            resnet_dict = self.resnet.state_dict()
            resnet_dict.update(init)
            self.resnet.load_state_dict(resnet_dict)

        self.resnet = nn.DataParallel(self.resnet)

        ###########################
        #                         #
        # DO NOT USE FC Layer !!! #
        #                         #
        ###########################
        # self.resnet = nn.DataParallel(self.resnet)
        # if pretrained==True:
        #     self.resnet.try_to_load_state_dict = types.MethodType(try_to_load_state_dict,self.resnet)
        #     self.resnet.try_to_load_state_dict(torch.load(savefile))
        #
        # self.fc1 = nn.Linear(2048,1024)
        # self.fc1 = nn.DataParallel(self.fc1)
        #
        # self.relu1 = nn.ReLU()
        # self.relu1 = nn.DataParallel(self.relu1)
        #
        # self.drop1 = nn.Dropout(dropout)
        # self.drop1 = nn.DataParallel(self.drop1)
        #
        # self.fc2 = nn.Linear(1024,101)
        # self.fc2 = nn.DataParallel(self.fc2)

    def forward(self, x):

        x = self.resnet(x)

        # x = self.fc1(x)
        # x = self.relu1(x)
        # x = self.drop1(x)
        # x = self.fc2(x)

        return x

    def inference(self, x):

        return self.forward(x)

        # x = self.resnet(x)
        # self.midfeatures = x
        # x = self.fc1(x)
        # x = self.relu1(x)
        # x = self.fc2(x)
        # return x


def resnet_SpatialNet(pretrained=False, level=101, **kwargs):
    return resnet_TwoStreamNet(in_channels=3, pretrained=pretrained, level=level, **kwargs)


def resnet_TemporalNet(pretrained=False, level=101, **kwargs):
    return resnet_TwoStreamNet(in_channels=20, pretrained=pretrained, level=level, **kwargs)


def resnet101_SpatialNet(pretrained=False, **kwargs):
    return resnet_SpatialNet(pretrained=pretrained, level=101, **kwargs)


def resnet101_TemporalNet(pretrained=False, **kwargs):
    return resnet_TemporalNet(pretrained=pretrained, level=101, **kwargs)


def resnet152_SpatialNet(pretrained=False, **kwargs):
    return resnet_SpatialNet(pretrained=pretrained, level=152, **kwargs)


def resnet152_TemporalNet(pretrained=False, **kwargs):
    return resnet_TemporalNet(pretrained=pretrained, level=152, **kwargs)


def resnet50_TemporalNet(pretrained=False, **kwargs):
    return resnet_TemporalNet(pretrained=pretrained, level=50, **kwargs)


if __name__ == '__main__':
    model = resnet50_TemporalNet(False).cuda()
    x = torch.randn(2, 20, 224, 224)
    x = Variable(x).cuda()

    y = model.inference(x)
    y = model.inference(x)
    pass
