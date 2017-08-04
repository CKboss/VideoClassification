import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from torchvision.models import densenet161
from torchvision.models import inception_v3

import numpy as np
from VideoClassification.utils.toolkits import accuracy


out = np.array([[0.1,-0.1,0],[0.1,-0.2,0.3],[1,-0.1,9]])

target = np.array([1,2,0])

out = Variable(torch.from_numpy(out))
target = Variable(torch.from_numpy(target))

acc = accuracy(out,target,topk=(1,2))

acc = [ a.data[0] for a in acc]