import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from torchvision.models import densenet161

def acc(output:Variable,target:Variable) ->float:
    nn.Softmax(output)
