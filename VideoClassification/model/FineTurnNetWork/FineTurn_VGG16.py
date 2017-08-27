import torch
import torch.nn as nn
from torch.autograd import Variable

from torchvision.models.vgg import vgg16
from torchvision.models.inception import inception_v3
from torchvision.models.resnet import resnet152


def FineTurn_VGG16():
    model = vgg16().cuda()
    new_classfier = list(model.classifier)[:-3]
    model.classifier = nn.Sequential(*new_classfier)
