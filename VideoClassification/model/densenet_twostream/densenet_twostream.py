import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import VideoClassification.Config.Config as Config
from VideoClassification.model.densenet_twostream.densenet import densenet161,densenet201

class dense_twostram(nn.Module):

    def __init__(self,pretrained=False,in_channels=3,num_classes=101,level=161,drop1=0.9,drop2=0.8):

        super(dense_twostram,self).__init__()

        if level==161:
            DENSEnet = densenet161
        elif level==201:
            DENSEnet = densenet201
        else:
            raise NotImplementedError('level should be 161 or 201')

        self.dense = DENSEnet(in_channels=in_channels)

        self.train_classification = nn.Sequential(
            nn.Linear(2208,1024),
            nn.ReLU(True),
            nn.Dropout(drop1),
            nn.Linear(1024,num_classes),
        )

        self.test_classfication = nn.Sequential (
            nn.Linear(2208,1024),
            nn.Linear(1024,num_classes),
        )

    # def forward(self,x):
    #     x = self.dense(x)
    #     x = self.train_classification(x)
    #     return x
    #
    # def inference(self,x):
    #     x = self.dense(x)
    #     x = self.test_classfication(x)
    #     return x

def dense_spatialNet(pretrained=False,**kwargs):
    return dense_twostram(pretrained=pretrained,in_channels=3,drop1=0.8,**kwargs)

def dense_temporalNet(pretrained=False,**kwargs):
    return dense_twostram(pretrained=pretrained,in_channels=20,drop1=0.9,**kwargs)

def dense161_spatialNet(pretrained=False):
    return dense_spatialNet(pretrained=pretrained,level=161)

def dense161_temporalNet(pretrained=False):
    return dense_temporalNet(pretrained=pretrained,level=161)

def dense201_spatialNet(pretrained=False):
    return dense_spatialNet(pretrained=pretrained,level=201)

def dense201_temporalNet(pretrained=False):
    return dense_temporalNet(pretrained=pretrained,level=201)

if __name__=='__main__':

    x = torch.randn(2,3,224,224)
    x = Variable(x).cuda()

    module = dense161_spatialNet().cuda()

    y = module(x)
    print(y.size())
