import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable

class C3D(nn.Module):

    def __init__(self,drop=0.8,num_classes=101):

        super(C3D,self).__init__()

        self.num_classes=101

        self.conv1 = nn.Conv3d(3,64,kernel_size=3,padding=1)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm3d(64)
        self.mxpool1 = nn.MaxPool3d((1,2,2))
        self.conv2 = nn.Conv3d(64,128,kernel_size=3,padding=1)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm3d(128)
        self.mxpool2 = nn.MaxPool3d(2)
        self.conv3 = nn.Conv3d(128,256,kernel_size=3,padding=1)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv3d(256,256,kernel_size=3,padding=1)
        self.relu4 = nn.ReLU()
        self.bn3 = nn.BatchNorm3d(256)
        self.mxpool3 = nn.MaxPool3d(2)
        self.conv5 = nn.Conv3d(256,512,kernel_size=3,padding=1)
        self.relu5 = nn.ReLU()
        self.conv6 = nn.Conv3d(512,512,kernel_size=3,padding=1)
        self.relu6 = nn.ReLU()
        self.bn4 = nn.BatchNorm3d(512)
        self.mxpool4 = nn.MaxPool3d(2)
        self.conv7 = nn.Conv3d(512,512,kernel_size=3,padding=1)
        self.relu7 = nn.ReLU()
        self.conv8 = nn.Conv3d(512,512,kernel_size=3,padding=1)
        self.relu8 = nn.ReLU()
        self.bn5 = nn.BatchNorm3d(512)
        self.mxpool5 = nn.MaxPool3d(2)
        self.fc1 = nn.Linear(512*9*1,self.num_classes)
        # self.relu_fc1 = nn.ReLU()
        # self.fc2 = nn.Linear(2048,2048)
        # self.relu_fc2 = nn.ReLU()
        # self.fc3 = nn.Linear(2048,101)

        # self.drop1 = nn.Dropout(drop)
        # self.drop2 = nn.Dropout(drop)

        self.Flow = nn.Sequential (
            self.conv1,
            self.relu1,
            self.bn1,
            self.mxpool1,
            self.conv2,
            self.relu2,
            # self.bn2,
            self.mxpool2,
            self.conv3,
            self.relu3,
            self.conv4,
            self.relu4,
            # self.bn3,
            self.mxpool3,
            self.conv5,
            self.relu5,
            self.conv6,
            self.relu6,
            # self.bn4,
            self.mxpool4,
            self.conv7,
            self.relu7,
            self.conv8,
            self.relu8,
            # self.bn5,
            self.mxpool5,
        )


        self.Flow = nn.DataParallel(self.Flow)
        # self.fc1 = nn.DataParallel(self.fc1)
        # self.fc2 = nn.DataParallel(self.fc2)

        self._initialize_weights()


    def forward(self,x):

        x = self.Flow(x)
        x = x.view(-1,512*9*1)
        x = self.fc1(x)
        # x = self.relu_fc1(x)
        # x = self.drop1(x)
        # x = self.fc2(x)
        # x = self.relu_fc2(x)
        # x = self.drop2(x)
        # x = self.fc3(x)

        return x

    def inference(self,x):

        return self.forward(x)

        # x = self.Flow(x)
        # x = x.view(-1,512*9*1)
        # x = self.fc1(x)
        # x = self.relu_fc1(x)
        # x = self.fc2(x)
        # x = self.relu_fc2(x)
        # self.midfeatures = x
        # x = self.fc3(x)

        # return x


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m,nn.Conv3d):
                init.xavier_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.xavier_normal(m.weight.data)
                m.bias.data.zero_()


if __name__=='__main__':

    module = C3D().cuda()

    x = torch.randn(2,3,20,112,112)
    x = Variable(x).cuda()

    y = module(x)
    y = module.inference(x)
    z = module.midfeatures
