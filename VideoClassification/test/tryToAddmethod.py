from VideoClassification.model.vgg_twostream.VGG16 import vgg16
from VideoClassification.utils.toolkits import try_to_load_state_dict

import types

module = vgg16()

module.m1 = types.MethodType(try_to_load_state_dict,module)


pt = '/home/lab/BackUp/pretrained/vgg16-397923af.pth'
import torch

module.m1(torch.load(pt))
