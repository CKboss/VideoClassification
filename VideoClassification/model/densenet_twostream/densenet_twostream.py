import types

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable

import VideoClassification.Config.Config as Config
from VideoClassification.model.densenet_twostream.densenet import densenet169,densenet201

class dense_twostram(nn.Module):

    def __init__(self,pretrained=False,in_channels=3,num_classes=101,level=169,dropout=0.9,drop2=0.8,**kwargs):

        super(dense_twostram,self).__init__()

        savepath = None
        if level==169:
            DENSEnet = densenet169
            savepath = Config.densenet169_pretrainfile
            # nn1 = 2208
        elif level==201:
            DENSEnet = densenet201
            savepath = Config.densenet201_pretrainfile
            # nn1 = 1920
        else:
            raise NotImplementedError('level should be 161 or 201')

        self.dense = DENSEnet(in_channels=in_channels,num_classes=num_classes)

        self._initialize_weights()
        if pretrained==True:

            init_data = torch.load(savepath)
            load_data = self.dense.state_dict()
            load_data_keys = load_data.keys()

            for key1 in list(init_data.keys()):
                if key1 in load_data_keys and init_data[key1].size() == load_data[key1].size():
                    # print('update:',key1)
                    load_data[key1] = init_data[key1]

            self.dense.state_dict().update(load_data)


        # self.fc1 = nn.Linear(nn1,1024)
        # self.relu = nn.ReLU()
        # self.fc2 =nn.Linear(1024,num_classes)
        #
        # self.train_classification = nn.Sequential(
        #     self.fc1,
        #     self.relu,
        #     nn.Dropout(dropout),
        #     self.fc2
        # )

        # self.test_classfication = nn.Sequential (
        #     self.fc1,
        #     self.relu,
        #     self.fc2,
        # )

        # self.dense = nn.DataParallel(self.dense)
        # self.train_classification = nn.DataParallel(self.train_classification)
        #

    def forward(self,x):
        x = self.dense(x)
        # x = self.train_classification(x)
        return x

    def inference(self,x):
        return self.forward(x)
        # x = self.dense(x)
        # x = self.fc1(x)
        # x = self.relu(x)
        # self.midfeatures = x
        # x = self.fc2(x)
        # return x


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.xavier_normal(m.weight.data)
                m.bias.data.zero_()

def dense_spatialNet(pretrained=False,**kwargs):
    return dense_twostram(pretrained=pretrained,in_channels=3,drop1=0.8,**kwargs)

def dense_temporalNet(pretrained=False,**kwargs):
    return dense_twostram(pretrained=pretrained,in_channels=20,drop1=0.9,**kwargs)

def dense169_spatialNet(pretrained=False,**kwargs):
    return dense_spatialNet(pretrained=pretrained,level=169,**kwargs)

def dense169_temporalNet(pretrained=False,**kwargs):
    return dense_temporalNet(pretrained=pretrained,level=169,**kwargs)

def dense201_spatialNet(pretrained=False,**kwargs):
    return dense_spatialNet(pretrained=pretrained,level=201,**kwargs)

def dense201_temporalNet(pretrained=False,**kwargs):
    return dense_temporalNet(pretrained=pretrained,level=201,**kwargs)

if __name__=='__main__':

    x = torch.randn(2,20,224,224)
    x = Variable(x).cuda()

    module = dense201_spatialNet(True).cuda()

    y = module(x)
    print(y.size())
