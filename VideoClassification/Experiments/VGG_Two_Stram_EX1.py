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
from VideoClassification.utils.data_pretreatment.PipeLine import ImgAugPipes,GenTensors
from VideoClassification.utils.toolkits import accuracy

'''
VGG TWO Stram 测试:
1. Spatial 输入单张图片, SGD 10个 epoch 每个迭代4000次 
初始学习率0.001, 每个epoch学习率*0.1

2. Temporal 输入连续多20张光流, SGD 20个 epoch 每个迭代10000次
初始学习率0.005 每个epoch学习率*0.1
'''


############ Config

logger = Logger(Config.LOGSpace+'/EX1')
savepath = Config.ExWorkSpace+'/EX1/'
batchsize = 86

############


def GenVariables(dsl,**kwargs):

    # TODO add requires_grad params

    items = random.choices(dsl,k=batchsize)

    imgpathss = []
    labels = []

    for item in items:
        imgpathss.append(item[0])
        labels.append(item[1])

    imgs = GenTensors(imgpathss,isTemporal=True)

    imgs = Variable(imgs,**kwargs).float().cuda()
    labels = Variable(torch.from_numpy(np.array(labels)),**kwargs).long().cuda()

    return imgs,labels

def GenVariables_Spatial(dsl,**kwargs):

    # TODO add requires_grad params

    items = random.choices(dsl,k=batchsize)

    imgpaths = []
    labels = []

    for item in items:
        imgpaths.append(item[0])
        labels.append(item[1])

    imgs = []

    for path in imgpaths:
        imgs.append(cv2.imread(path))

    imgs = np.array(ImgAugPipes(imgs))
    imgs = Variable(torch.from_numpy(imgs),**kwargs).float().cuda()
    labels = Variable(torch.from_numpy(np.array(labels)),**kwargs).long().cuda()

    return imgs,labels


def VGG_Temporal_Net_Run():

    epochs = 20
    loops = 10000
    learningrate = 0.005
    attenuation = 0.1

    train_dsl = train_UCF0101_Temporal()
    test_dsl = test_UCF0101_Temporal()

    model = VGG_Temporal_Net(pretrained=True).cuda()
    lossfunc = nn.CrossEntropyLoss()
    optim = torch.optim.SGD(model.parameters(),lr=learningrate)

    cnt = 0
    for epoch in range(epochs) :

        for l in range(loops) :

            cnt+=1

            imgs,labels = GenVariables(train_dsl)

            model.zero_grad()
            pred =  model(imgs)
            loss = lossfunc(pred,labels)

            logger.scalar_summary('Temporal/train_loss',loss.data[0],cnt)

            loss.backward()
            optim.step()


            print('Temporal epoch: {} cnt: {}'.format(epoch,cnt))

            if cnt%50 == 0:

                imgs,labels = GenVariables(test_dsl)
                pred = model.inference(imgs)
                loss = lossfunc(pred,labels)

                logger.scalar_summary('Temporal/test_loss',loss.data[0],cnt)

                #acc
                acc = accuracy(pred,labels,topk=(1,5,10))
                logger.scalar_summary('Temporal/test_acc@1',acc[0],cnt)
                logger.scalar_summary('Temporal/test_acc@5',acc[1],cnt)
                logger.scalar_summary('Temporal/test_acc@10',acc[2],cnt)


                imgs,labels = GenVariables(train_dsl)
                pred = model.inference(imgs)

                acc = accuracy(pred,labels,topk=(1,5,10))
                logger.scalar_summary('Temporal/train_acc@1',acc[0],cnt)
                logger.scalar_summary('Temporal/train_acc@5',acc[5],cnt)
                logger.scalar_summary('Temporal/train_acc@10',acc[10],cnt)


        learningrate = learningrate*attenuation
        optim = torch.optim.SGD(model.parameters(),lr=learningrate)

        savefile = savepath + 'VGG_Temporal_EX1_{:02d}.pt'.format(epoch%100)
        print('Temporal save model to {}'.format(savefile))
        torch.save(model,savefile)


def VGG_Spatial_Net_Run():

    epochs = 10
    loops = 4000
    learningrate = 0.001
    attenuation = 0.1

    train_dsl = train_UCF0101_Spatial()
    test_dsl = test_UCF0101_Spatial()

    model = VGG_Spatial_Net(pretrained=True).cuda()
    lossfunc = nn.CrossEntropyLoss()
    optim = torch.optim.SGD(model.parameters(),lr=learningrate)

    cnt = 0

    for epoch in range(epochs) :

        for l in range(loops) :

            cnt+=1

            imgs,labels = GenVariables_Spatial(train_dsl)

            model.zero_grad()
            pred =  model(imgs)
            loss = lossfunc(pred,labels)

            logger.scalar_summary('Spatial/train_loss',loss.data[0],cnt)

            loss.backward()
            optim.step()

            print('Spatial epoch: {} cnt: {}'.format(epoch,cnt))

            if cnt%50 == 0:

                imgs,labels = GenVariables_Spatial(test_dsl)
                pred = model.inference(imgs)
                loss = lossfunc(pred,labels)
                logger.scalar_summary('Spatial/test_loss',loss.data[0],cnt)

                acc = accuracy(pred,labels,topk=(1,5,10))
                logger.scalar_summary('Spatial/test_acc@1',acc[0],cnt)
                logger.scalar_summary('Spatial/test_acc@5',acc[1],cnt)
                logger.scalar_summary('Spatial/test_acc@10',acc[2],cnt)

                imgs,labels = GenVariables_Spatial(train_dsl)
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
