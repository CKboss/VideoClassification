import torch
import torch.nn as nn
from torch.autograd import Variable

from VideoClassification.model.InceptionV3_twostream.InceptionV3 import inception_v3

model = inception_v3(num_classes=101)


class _InceptionV3(nn.Module):
    def __init__(self, inchannels=3, num_classes=101, drop1=0.8):
        super(_InceptionV3, self).__init__()

        self.num_classes = num_classes
        self.ince3 = inception_v3(num_classes=num_classes, inchannels=inchannels)
        self.drop1 = nn.Dropout(drop1)
        self.fc2 = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.ince3(x)
        x = self.drop1(x)
        x = self.fc2(x)

        return x

    def inference(self, x):
        x = self.ince3(x)
        self.midfeatures = x
        x = self.fc2(x)

        return x


def get_Spatial_Net(**kwargs):
    return _InceptionV3(inchannels=3, **kwargs)


def get_Temporal_Net(**kwargs):
    return _InceptionV3(inchannels=20, **kwargs)


if __name__ == '__main__':
    x = torch.randn(3, 3, 299, 299)
    x = Variable(x)
    model = get_Spatial_Net()
    y = model(x)
