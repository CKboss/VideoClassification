import numpy as np
import torch
from torch.autograd import Variable

from VideoClassification.utils.Others.toolkits import accuracy

out = np.array([[0.1,-0.1,0],[0.1,-0.2,0.3],[1,-0.1,9]])

target = np.array([1,2,0])

out = Variable(torch.from_numpy(out))
target = Variable(torch.from_numpy(target))

acc = accuracy(out,target,topk=(1,2))

acc = [ a.data[0] for a in acc]

import torchvision.transforms as transforms

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
