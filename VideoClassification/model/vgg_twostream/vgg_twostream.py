import math
import types

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import init

import VideoClassification.Config.Config as Config
from VideoClassification.model.vgg_twostream.VGG16 import vgg16
from VideoClassification.utils.Others.toolkits import try_to_load_state_dict

class VGG_Temporal_Net(nn.Module):

    def __init__(self,pretrained=False,dropout1=0.9,dropout2=0.8,num_classes=101):

        super(VGG_Temporal_Net,self).__init__()

        self.num_classes = num_classes
        self.vgg16 = vgg16(in_channels=20,num_classes=self.num_classes)
        self.vgg16 = nn.DataParallel(self.vgg16)

        self.fc1 = nn.Linear(512*7*7,4096)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(4096,4096)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(4096,101)

        self.train_classifier = nn.Sequential(
            self.fc1,
            self.relu1,
            nn.Dropout(dropout1),
            self.fc2,
            self.relu2,
            nn.Dropout(dropout2),
            self.fc3,
        )

        self.train_classifier = nn.DataParallel(self.train_classifier)
        self.init_weight()

        if pretrained==True:
            # self.vgg16.try_to_load_state_dict = types.MethodType(try_to_load_state_dict,self.vgg16)
            # self.vgg16.try_to_load_state_dict(torch.load(Config.vgg16pretrainfile))
            init_data = torch.load(Config.vgg16pretrainfile)
            origin_data = self.vgg16.state_dict()
            tmp_data = dict()

            for key in init_data.keys():
                newkey = 'module.'+key
                if newkey in origin_data.keys() and init_data[key].size() == origin_data[newkey].size() :
                    tmp_data[newkey] = init_data[key]
            self.vgg16.state_dict().update(origin_data)


    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m,nn.Linear):
                init.xavier_uniform(m.weight.data)

    def forward(self,x):
        x = self.vgg16(x)
        x = self.train_classifier(x)
        return x

    def inference(self,x):
        return self.forward(x)

class VGG_Spatial_Net(nn.Module):

    def __init__(self,pretrained=False,dropout1=0.9,dropout2=0.8,num_classes=101):

        super(VGG_Spatial_Net,self).__init__()

        self.num_classes = num_classes

        self.vgg16 = vgg16(in_channels=3,num_classes=self.num_classes)
        self.vgg16 = nn.DataParallel(self.vgg16)

        self.fc1 = nn.Linear(512*7*7,4096)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(4096,4096)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(4096,self.num_classes)


        self.train_classifier = nn.Sequential(
            self.fc1,
            self.relu1,
            nn.Dropout(dropout1),
            self.fc2,
            self.relu2,
            nn.Dropout(dropout2),
            self.fc3,
        )

        self.train_classifier = nn.DataParallel(self.train_classifier)

        self.init_weight()
        if pretrained==True:
            # self.vgg16.try_to_load_state_dict = types.MethodType(try_to_load_state_dict,self.vgg16)
            # self.vgg16.try_to_load_state_dict(torch.load(Config.vgg16pretrainfile))
            init_data = torch.load(Config.vgg16pretrainfile)
            origin_data = self.vgg16.state_dict()
            tmp_data = dict()

            for key in init_data.keys():
                newkey = 'module.'+key
                if newkey in origin_data.keys() and init_data[key].size() == origin_data[newkey].size() :
                    tmp_data[newkey] = init_data[key]
            self.vgg16.state_dict().update(origin_data)

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m,nn.Linear):
                init.xavier_uniform(m.weight.data)

    def forward(self,x):
        x = self.vgg16(x)
        x = self.train_classifier(x)
        return x

    def inference(self,x):
        return self.forward(x)

 # try to merge spatial and temporal

if __name__=='__main__':


    model = VGG_Temporal_Net(pretrained=True).cuda()
    # pt = '/home/lab/BackUp/pretrained/vgg16-397923af.pth'
    # vgg16.load_state_dict(torch.load(pt))


    x = torch.randn(2,3,224,224)
    x = Variable(x).cuda()
    model = VGG_Spatial_Net(pretrained=True).cuda()
    y = model.inference(x)
    z = model.midfeatures

    print(y.size())
    print(z.size())
