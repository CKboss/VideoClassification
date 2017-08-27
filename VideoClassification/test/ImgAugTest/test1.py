import time
import torch
import torch.nn as nn
from torch.autograd import Variable

try:
    from cv2 import cv2
except:
    import cv2

import VideoClassification.Config.Config as Config
from VideoClassification.model.resnet_twostream.resnet_twostream import resnet152_SpatialNet, resnet101_SpatialNet
from VideoClassification.utils.Others.Logger import Logger
from VideoClassification.utils.DataSetLoader.UCF101_DataSetLoader_FromFileName.UCF101Loader import \
    train_UCF0101_Spatial, test_UCF0101_Spatial, train_UCF0101_Temporal, test_UCF0101_Temporal, train_UCF101_C3D, \
    test_UCF101_C3D
from VideoClassification.utils.Others.toolkits import accuracy, try_to_load_state_dict
from VideoClassification.utils.DataSetLoader.UCF101_DataSetLoader_FromFileName.PictureQueue import PictureQueue, \
    GenVariables_Spatial, GenVariables_Temporal, GenVariables_C3D

batchsize = 86

epochs = 80
loops = 2000

TEST = 'S'

if TEST == 'T':
    pq_train = PictureQueue(dsl=train_UCF0101_Temporal(), Gen=GenVariables_Temporal, batchsize=batchsize)
    pq_test = PictureQueue(dsl=test_UCF0101_Temporal(), Gen=GenVariables_Temporal, batchsize=batchsize)
elif TEST == 'S':
    pq_train = PictureQueue(dsl=train_UCF0101_Spatial(), Gen=GenVariables_Spatial, batchsize=batchsize)
    pq_test = PictureQueue(dsl=test_UCF0101_Spatial(), Gen=GenVariables_Spatial, batchsize=batchsize)
elif TEST == 'C3D':
    pq_train = PictureQueue(dsl=train_UCF101_C3D(), Gen=GenVariables_C3D, batchsize=batchsize)
    pq_test = PictureQueue(dsl=test_UCF101_C3D(), Gen=GenVariables_C3D, batchsize=batchsize)

print('before:', time.asctime())
a, b = GenVariables_Temporal(train_UCF0101_Temporal(), batchsize=batchsize)
print('after:', time.asctime())

cnt = 0

for epoch in range(epochs):
    for l in range(loops):
        cnt += 1
        print('{}: {}'.format(time.asctime(), cnt))
        imgs, labels = pq_train.Get()
        if cnt % 25 == 0:
            imgs, labels = pq_test.Get()
            imgs, labels = pq_train.Get()
