import time
import torch
import torch.nn as nn
from torch.autograd import Variable

try:
    from cv2 import cv2
except:
    import cv2

import VideoClassification.Config.Config as Config
from VideoClassification.model.resnet_twostream.resnet_twostream import resnet152_SpatialNet,resnet101_SpatialNet,resnet152_TemporalNet,resnet50_TemporalNet,resnet101_TemporalNet
from VideoClassification.model.densenet_twostream.densenet_twostream import dense201_spatialNet,dense201_temporalNet,dense169_temporalNet,dense169_spatialNet
from VideoClassification.model.C3D.C3D import C3D
from VideoClassification.utils.Others.Logger import Logger
from VideoClassification.utils.DataSetLoader.UCF101_DataSetLoader_FromFileName.UCF101Loader import train_UCF0101_Spatial,test_UCF0101_Spatial
from VideoClassification.utils.Others.toolkits import accuracy,try_to_load_state_dict
from VideoClassification.utils.DataSetLoader.UCF101_DataSetLoader_FromFileName.PictureQueue import PictureQueue,GenVariables_Spatial


def Resenet152_SpatialNet_TestRuntime_Run():

    model = dense201_spatialNet().cuda()

    for batchsize in [4,8,16,24,32]:

        total_gen = 0
        total_batch = 0


        for i in range(15):

            t2 = time.time()

            # imgs,labels = pq_train.Get()

            imgs = torch.randn(batchsize,3,224,224)
            imgs = Variable(imgs).cuda(0)

            t3 = time.time()

            pred =  model(imgs)

            t4 = time.time()

            total_gen += t3-t2
            total_batch += t4-t3

        print('batchsize:',batchsize)
        print('gen_data:',total_gen/30)
        print('one batch',total_batch/30)

