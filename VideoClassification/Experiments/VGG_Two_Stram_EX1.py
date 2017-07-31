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
from VideoClassification.model.vgg_twostream.vgg_twostream import VGG_Temporal_Net,VGG_Spatial_Net
from VideoClassification.utils.Logger import Logger
from VideoClassification.utils.DataSetLoader.UCF101Loader import train_UCF0101_Temporal,test_UCF0101_Temporal,train_UCF0101_Spatial,test_UCF0101_Spatial
from VideoClassification.utils.toolkits import accuracy
from VideoClassification.utils.DataSetLoader.PictureQueue import PictureQueue,GenVariables_Spatial,GenVariables_Temporal

'''
VGG TWO Stram 测试:
1. Spatial 输入单张图片, SGD 10个 epoch 每个迭代4000次 
初始学习率0.001, 每个epoch学习率*0.1

2. Temporal 输入连续多20张光流, SGD 20个 epoch 每个迭代5000次
初始学习率0.05 每个epoch学习率*0.5
'''


############ Config

logger = Logger(Config.LOGSpace+'/EX2')
savepath = Config.ExWorkSpace+'/EX2/'

import os.path
if os.path.isdir(savepath)==False:
    os.mkdir(savepath)

batchsize = 86

############


def VGG_Temporal_Net_Run():

    epochs = 20
    loops = 5000
    learningrate = 0.1
    attenuation = 0.5

    model = VGG_Temporal_Net(pretrained=True).cuda()
    lossfunc = nn.CrossEntropyLoss()
    optim = torch.optim.SGD(model.parameters(),lr=learningrate,momentum=0.001)

    pq_train = PictureQueue(dsl=train_UCF0101_Temporal(),Gen=GenVariables_Temporal,batchsize=batchsize)
    pq_test = PictureQueue(dsl=test_UCF0101_Temporal(),Gen=GenVariables_Temporal,batchsize=batchsize)

    cnt = 0
    for epoch in range(epochs) :

        for l in range(loops) :

            cnt+=1

            imgs,labels = pq_train.Get()

            model.zero_grad()
            pred =  model(imgs)
            loss = lossfunc(pred,labels)

            logger.scalar_summary('Temporal/train_loss',loss.data[0],cnt)

            loss.backward()
            optim.step()


            print('Temporal epoch: {} cnt: {}'.format(epoch,cnt))

            if cnt%50 == 0:

                imgs,labels = pq_test.Get()
                pred = model.inference(imgs)
                loss = lossfunc(pred,labels)

                logger.scalar_summary('Temporal/test_loss',loss.data[0],cnt)

                #acc
                acc = accuracy(pred,labels,topk=(1,5,10))
                logger.scalar_summary('Temporal/test_acc@1',acc[0],cnt)
                logger.scalar_summary('Temporal/test_acc@5',acc[1],cnt)
                logger.scalar_summary('Temporal/test_acc@10',acc[2],cnt)


                imgs,labels = pq_train.Get()
                pred = model.inference(imgs)

                acc = accuracy(pred,labels,topk=(1,5,10))
                logger.scalar_summary('Temporal/train_acc@1',acc[0],cnt)
                logger.scalar_summary('Temporal/train_acc@5',acc[1],cnt)
                logger.scalar_summary('Temporal/train_acc@10',acc[2],cnt)


        learningrate = learningrate*attenuation
        optim = torch.optim.SGD(model.parameters(),lr=learningrate,momentum=0.001)

        savefile = savepath + 'VGG_Temporal_EX1_{:02d}.pt'.format(epoch%100)
        print('Temporal save model to {}'.format(savefile))
        torch.save(model,savefile)


def VGG_Spatial_Net_Run():

    epochs = 20
    loops = 4000
    learningrate = 0.1
    attenuation = 0.1

    model = VGG_Spatial_Net(pretrained=True).cuda()
    lossfunc = nn.CrossEntropyLoss()
    optim = torch.optim.SGD(model.parameters(),lr=learningrate)

    cnt = 0

    pq_train = PictureQueue(dsl=train_UCF0101_Spatial(),Gen=GenVariables_Spatial,batchsize=batchsize)
    pq_test = PictureQueue(dsl=test_UCF0101_Spatial(),Gen=GenVariables_Spatial,batchsize=batchsize)

    for epoch in range(epochs) :

        for l in range(loops) :

            cnt+=1

            imgs,labels = pq_train.Get()

            model.zero_grad()
            pred =  model(imgs)
            loss = lossfunc(pred,labels)

            logger.scalar_summary('Spatial/train_loss',loss.data[0],cnt)

            loss.backward()
            optim.step()

            print('Spatial epoch: {} cnt: {}'.format(epoch,cnt))

            if cnt%50 == 0:

                imgs,labels = pq_test.Get()
                pred = model.inference(imgs)

                loss = lossfunc(pred,labels)
                logger.scalar_summary('Spatial/test_loss',loss.data[0],cnt)

                acc = accuracy(pred,labels,topk=(1,5,10))
                logger.scalar_summary('Spatial/test_acc@1',acc[0],cnt)
                logger.scalar_summary('Spatial/test_acc@5',acc[1],cnt)
                logger.scalar_summary('Spatial/test_acc@10',acc[2],cnt)

                imgs,labels = pq_train.Get()
                pred = model.inference(imgs)

                acc = accuracy(pred,labels,topk=(1,5,10))
                logger.scalar_summary('Spatial/train_acc@1',acc[0],cnt)
                logger.scalar_summary('Spatial/train_acc@5',acc[1],cnt)
                logger.scalar_summary('Spatial/train_acc@10',acc[2],cnt)


        learningrate = learningrate*attenuation
        optim = torch.optim.SGD(model.parameters(),lr=learningrate)

        savefile = savepath + 'VGG_Spatial_EX1_{:02d}.pt'.format(epoch%100)
        print('Spatial save model to {}'.format(savefile))
        torch.save(model,savefile)


def DoubleRun():
    pool = Pool(processes=2)
    pool.apply_async(VGG_Temporal_Net_Run,())
    pool.apply_async(VGG_Spatial_Net_Run,())
    pool.close()
    pool.join()


if __name__=='__main__':
    # VGG_Temporal_Net_Run()
    pass
