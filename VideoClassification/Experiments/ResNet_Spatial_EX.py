import random
from multiprocessing import Pool

import torch
import torch.nn as nn
from torch.autograd import Variable

try:
    from cv2 import cv2
except:
    import cv2

import numpy as np


import VideoClassification.Config.Config as Config
from VideoClassification.model.resnet_twostream.resnet_twostream import resnet152_SpatialNet,resnet101_SpatialNet
from VideoClassification.utils.Logger import Logger
from VideoClassification.utils.DataSetLoader.UCF101Loader import train_UCF0101_Temporal,test_UCF0101_Temporal,train_UCF0101_Spatial,test_UCF0101_Spatial
from VideoClassification.utils.toolkits import accuracy,try_to_load_state_dict
from VideoClassification.utils.DataSetLoader.PictureQueue import PictureQueue,GenVariables_Spatial,GenVariables_Temporal


if __name__=='__main__':
    pass
