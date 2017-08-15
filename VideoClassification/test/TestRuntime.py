import time
import torch
import torch.nn as nn
from torch.autograd import Variable

try:
    from cv2 import cv2
except:
    import cv2

import VideoClassification.Config.Config as Config
from VideoClassification.model.resnet_twostream.resnet_twostream import resnet152_SpatialNet,resnet101_SpatialNet
from VideoClassification.utils.Others.Logger import Logger
from VideoClassification.utils.DataSetLoader.UCF101_DataSetLoader_FromFileName.UCF101Loader import train_UCF0101_Spatial,test_UCF0101_Spatial
from VideoClassification.utils.Others.toolkits import accuracy,try_to_load_state_dict
from VideoClassification.utils.DataSetLoader.UCF101_DataSetLoader_FromFileName.PictureQueue import PictureQueue,GenVariables_Spatial


def Resenet152_SpatialNet_TestRuntime_Run():

    model = resnet152_SpatialNet(pretrained=True,dropout=0.95).cuda()

    batchsize=32

    t1 = time.time()

    pq_train = PictureQueue(dsl=train_UCF0101_Spatial(),Gen=GenVariables_Spatial,batchsize=batchsize)

    t2 = time.time()

    print('prepare_pq:',t2-t1)

    for i in range(30):

        t2 = time.time()

        imgs,labels = pq_train.Get()

        t3 = time.time()

        pred =  model(imgs)

        t4 = time.time()

        print('gen_data:',t3-t2)
        print('one batch',t4-t3)

