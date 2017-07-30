import types

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from VideoClassification.model.vgg_twostream.VGG16 import vgg16
from VideoClassification.utils.toolkits import try_to_load_state_dict
import VideoClassification.Config.Config as Config


class VGG_Temporal_Net(nn.Module):

    def __init__(self,pretrained=False):

        super(VGG_Temporal_Net,self).__init__()

        self.vgg16 = vgg16(in_channels=20,num_classes=101)
        self.vgg16 = nn.DataParallel(self.vgg16)

        if pretrained==True:
            self.vgg16.try_to_load_state_dict = types.MethodType(try_to_load_state_dict,self.vgg16)
            self.vgg16.try_to_load_state_dict(torch.load(Config.vgg16pretrainfile))

        self.train_classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(0.9),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.8),
            nn.Linear(4096, 101),
        )

        self.eval_classifier_1 = nn.Sequential (
            nn.Linear(512*7*7,4096),
            nn.ReLU(True),
            nn.Linear(4096, 4096),
        )

        self.eval_classifier_2 = nn.Sequential(
            nn.ReLU(True),
            nn.Linear(4096, 101),
        )

    def forward(self,x):
        x = self.vgg16(x)
        x = self.train_classifier(x)
        return x

    def inference(self,x):
        x = self.vgg16(x)
        x = self.eval_classifier_1(x)
        self.midfeatures = x
        x = self.eval_classifier_2(x)
        return x

class VGG_Spatial_Net(nn.Module):

    def __init__(self,pretrained=False):

        super(VGG_Spatial_Net,self).__init__()

        self.vgg16 = vgg16(in_channels=3,num_classes=101)
        self.vgg16 = nn.DataParallel(self.vgg16)

        if pretrained==True:
            self.vgg16.try_to_load_state_dict = types.MethodType(try_to_load_state_dict,self.vgg16)
            self.vgg16.try_to_load_state_dict(torch.load(Config.vgg16pretrainfile))

        self.train_classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(0.95),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.9),
            nn.Linear(4096, 101),
        )

        self.eval_classifier_1 = nn.Sequential (
            nn.Linear(512*7*7,4096),
            nn.ReLU(True),
            nn.Linear(4096, 4096),
        )

        self.eval_classifier_2 = nn.Sequential(
            nn.ReLU(True),
            nn.Linear(4096, 101),
        )

    def forward(self,x):
        x = self.vgg16(x)
        x = self.train_classifier(x)
        return x

    def inference(self,x):
        x = self.vgg16(x)
        x = self.eval_classifier_1(x)
        self.midfeatures = x
        x = self.eval_classifier_2(x)
        return x

 # try to merge spatial and temporal

if __name__=='__main__':

    pass

    vgg16 = vgg16(in_channels=20,num_classes=101).cuda()
    pt = '/home/lab/BackUp/pretrained/vgg16-397923af.pth'
    vgg16.load_state_dict(torch.load(pt))

    x = torch.randn(2,20,224,224)
    x = Variable(x).cuda()
    module = VGG_Temporal_Net(pretrained=True).cuda()
    y = module.inference(x)
    z = module.midfeatures
