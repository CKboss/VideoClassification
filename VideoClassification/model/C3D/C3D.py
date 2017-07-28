import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class C3D(nn.Module):

    def __init__(self):

        super(C3D,self).__init__()

        self.conv1 = nn.Conv3d(3,64,kernel_size=3,padding=1)
        self.mxpool1 = nn.MaxPool3d((1,2,2))
        self.conv2 = nn.Conv3d(64,128,kernel_size=3,padding=1)
        self.mxpool2 = nn.MaxPool3d(2)
        self.conv3 = nn.Conv3d(128,256,kernel_size=3,padding=1)
        self.conv4 = nn.Conv3d(256,256,kernel_size=3,padding=1)
        self.mxpool3 = nn.MaxPool3d(2)
        self.conv5 = nn.Conv3d(256,512,kernel_size=3,padding=1)
        self.conv6 = nn.Conv3d(512,512,kernel_size=3,padding=1)
        self.mxpool4 = nn.MaxPool3d(2)
        self.conv7 = nn.Conv3d(512,512,kernel_size=3,padding=1)
        self.conv8 = nn.Conv3d(512,512,kernel_size=3,padding=1)
        self.mxpool5 = nn.MaxPool3d(2)
        self.fc1 = nn.Linear(512*9,4096)
        self.fc2 = nn.Linear(4096,101)


    def forward(self,x):

        x = self.conv1(x)
        x = self.mxpool1(x)
        x = self.conv2(x)
        x = self.mxpool2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.mxpool3(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.mxpool4(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.mxpool5(x)

        x = x.view(-1,512*9)

        x = self.fc1(x)
        self.midfeatures = x
        x = self.fc2(x)

        return x


if __name__=='__main__':

    module = C3D().cuda()

    x = torch.randn(2,3,16,112,112)
    x = Variable(x).cuda()

    y = module(x)
    z = module.midfeatures
