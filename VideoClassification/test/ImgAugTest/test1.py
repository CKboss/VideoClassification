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
from VideoClassification.utils.DataSetLoader.UCF101_DataSetLoader_FromFileName.UCF101Loader import train_UCF0101_Spatial,test_UCF0101_Spatial,train_UCF0101_Temporal,test_UCF0101_Temporal
from VideoClassification.utils.Others.toolkits import accuracy,try_to_load_state_dict
from VideoClassification.utils.DataSetLoader.UCF101_DataSetLoader_FromFileName.PictureQueue import PictureQueue,GenVariables_Spatial,GenVariables_Temporal



batchsize = 86


epochs = 80
loops = 2000


pq_train = PictureQueue(dsl=train_UCF0101_Temporal(),Gen=GenVariables_Temporal,batchsize=batchsize)
pq_test = PictureQueue(dsl=test_UCF0101_Temporal(),Gen=GenVariables_Temporal,batchsize=batchsize,worker=10)

# pq_train = PictureQueue(dsl=train_UCF0101_Spatial(),Gen=GenVariables_Spatial,batchsize=batchsize,worker=5)
# pq_test = PictureQueue(dsl=test_UCF0101_Spatial(),Gen=GenVariables_Spatial,batchsize=batchsize,worker=2)

cnt = 0
for epoch in range(epochs) :

    for l in range(loops) :

        cnt+=1

        print('{}: {}'.format(time.asctime(),cnt))

        imgs,labels = pq_train.Get()

        if cnt%25 == 0:

            imgs,labels = pq_test.Get()

            imgs,labels = pq_train.Get()

