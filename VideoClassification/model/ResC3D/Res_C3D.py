import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable

class Bottleneck(nn.Module):

    expansion = 4

    def __init__(self,inplants,plants,downsample=None):

        super(Bottleneck,self).__init__()

        self.conv1 = nn.Conv3d(in_channels=inplants,out_channels=plants,kernel_size=3,padding=1)
        self.bn1 = nn.BatchNorm3d(plants)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(in_channels=plants,out_channels=plants,kernel_size=3,padding=1)
        self.bn2 = nn.BatchNorm3d(plants)

        self.conv3 = nn.Conv3d(in_channels=plants,out_channels=plants*self.expansion,kernel_size=3,padding=1)
        self.bn3 = nn.BatchNorm3d(plants*self.expansion)

        self.downsample = downsample

    def forward(self,x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # Get Res
        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out

class Res_C3D(nn.Module):

    def __init__(self,block,layers,inchannels=3,num_classes=101):

        super(Res_C3D,self).__init__()

        self.num_classes = num_classes
        self.inplanes = 64

        self.conv1 = nn.Conv3d(inchannels,64,kernel_size=3,padding=1)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(2)

        '''build layer'''
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AvgPool3d(kernel_size=7)

        self.Flow = nn.Sequential(
            self.conv1,
            self.bn1,
            self.relu,
            self.maxpool,

            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,

            self.avgpool,
        )

        self.Flow = nn.DataParallel(self.Flow)

        self.fc1 = nn.Linear(2048*1*8*8,2048)
        self.drop = nn.Dropout()
        self.fc2 = nn.Linear(2048,self.num_classes)

        self.fc1 = nn.DataParallel(self.fc1)
        self.fc2 = nn.DataParallel(self.fc2)

        self._initialize_weights()



    def forward(self,x):
        x = self.Flow(x)
        # 2048*1*8*8
        x = x.view(x.size(0),-1)
        x = self.fc1(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x

    def inference(self,x):
        x = self.Flow(x)
        x = x.view(x.size(0),-1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def _make_layer(self,block,planes,blocks):

        downsample = None

        if self.inplanes != planes*block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes,planes*block.expansion,kernel_size=3,padding=1),
                nn.BatchNorm2d(planes*block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes,planes,downsample))
        self.inplanes = planes*block.expansion
        for i in range(1,blocks):
            layers.append(block(self.inplanes,planes))

        return nn.Sequential(*layers)


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

def Res_C3D_1(**kwargs):
    return Res_C3D(Bottleneck,[1,1,1,1],**kwargs)

def Res_C3D_2(**kwargs):
    return Res_C3D(Bottleneck,[2,2,3,2],**kwargs)

if __name__=='__main__':

    x = torch.randn(1,3,20,112,112)
    x = Variable(x).cuda()

    downsample = nn.Sequential(
        nn.Conv3d(3,256,kernel_size=3,padding=1),
        nn.BatchNorm2d(256)
    )

    model = Bottleneck(3,64,downsample).cuda()

    c3d = Res_C3D(Bottleneck,[1,1,1,1]).cuda()

    y = c3d(x)

    y = model(x)

    print(x.size())
    print(y.size())

    z = downsample(x)

    print(z.size())
