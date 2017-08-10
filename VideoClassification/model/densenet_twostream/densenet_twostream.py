import types

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable

import VideoClassification.Config.Config as Config
from VideoClassification.model.densenet_twostream.densenet import densenet161,densenet201
from VideoClassification.utils.toolkits import try_to_load_state_dict

class dense_twostram(nn.Module):

    def __init__(self,pretrained=False,in_channels=3,num_classes=101,level=161,dropout=0.9,drop2=0.8,**kwargs):

        super(dense_twostram,self).__init__()

        savefile = None
        if level==161:
            DENSEnet = densenet161
            nn1 = 2208
        elif level==201:
            DENSEnet = densenet201
            nn1 = 1920
            savefile = Config.dense201_pretrainfile
        else:
            raise NotImplementedError('level should be 161 or 201')

        self.dense = DENSEnet(in_channels=in_channels)

        self.fc1 = nn.Linear(nn1,1024)
        self.relu = nn.ReLU()
        self.fc2 =nn.Linear(1024,num_classes)

        self.train_classification = nn.Sequential(
            self.fc1,
            self.relu,
            nn.Dropout(dropout),
            self.fc2
        )

        # self.test_classfication = nn.Sequential (
        #     self.fc1,
        #     self.relu,
        #     self.fc2,
        # )

        self.dense = nn.DataParallel(self.dense)
        self.train_classification = nn.DataParallel(self.train_classification)

        if pretrained==True:
            self.dense.try_to_load_state_dict = types.MethodType(try_to_load_state_dict,self.dense)
            self.dense.try_to_load_state_dict(torch.load(savefile))
            print('OK pretrained model: {} load success!'.format(savefile))
        elif pretrained==False:
            self._initialize_weights()

    def forward(self,x):
        x = self.dense(x)
        x = self.train_classification(x)
        return x

    def inference(self,x):
        x = self.dense(x)
        x = self.fc1(x)
        x = self.relu(x)
        self.midfeatures = x
        x = self.fc2(x)
        return x


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

def dense161_spatialNet(pretrained=False,**kwargs):
    return dense_spatialNet(pretrained=pretrained,level=161,**kwargs)

def dense161_temporalNet(pretrained=False,**kwargs):
    return dense_temporalNet(pretrained=pretrained,level=161,**kwargs)

def dense201_spatialNet(pretrained=False,**kwargs):
    return dense_spatialNet(pretrained=pretrained,level=201,**kwargs)

def dense201_temporalNet(pretrained=False,**kwargs):
    return dense_temporalNet(pretrained=pretrained,level=201,**kwargs)

if __name__=='__main__':

    x = torch.randn(2,20,224,224)
    x = Variable(x).cuda()

    module = dense201_temporalNet().cuda()

    y = module(x)
    print(y.size())
