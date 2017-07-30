import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
from torch.autograd import Variable

try:
    from cv2 import cv2
except:
    import cv2

import matplotlib.pyplot as plt
import numpy as np


import VideoClassification.Config.Config as Config
from VideoClassification.model.vgg_twostream.vgg_twostream import VGG_Temporal_Net,VGG_Spatial_Net
from VideoClassification.utils.Logger import Logger
from VideoClassification.utils.DataSetLoader.UCF101Loader import train_UCF0101_Temporal,test_UCF0101_Temporal,train_UCF0101_Spatial,test_UCF0101_Spatial
from VideoClassification.utils.data_pretreatment.PipeLine import ImgAugPipes,GenTensors
from VideoClassification.utils.toolkits import Accuracy


############ Config

logger = Logger(Config.LOGSpace+'/EX1')
savepath = Config.ExWorkSpace+'/EX1/'
batchsize = 8

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
    imgs = Variable(imgs,**kwargs).float().cuda()
    labels = Variable(torch.from_numpy(np.array(labels)),**kwargs).long().cuda()

    return imgs,labels


def VGG_Temporal_Net_Run():

    epochs = 20
    loops = 10000
    learningrate = 0.005
    attenuation = 0.1

    train_dsl = train_UCF0101_Temporal()
    test_dsl = test_UCF0101_Temporal()

    model = VGG_Temporal_Net(pretrained=False).cuda()
    lossfunc = nn.CrossEntropyLoss()
    optim = torch.optim.Adadelta(model.parameters(),lr=learningrate)

    cnt = 0
    for epoch in range(epochs) :

        for l in range(loops) :

            cnt+=1

            imgs,labels = GenVariables(train_dsl)

            model.zero_grad()
            pred =  model(imgs)
            loss = lossfunc(pred,labels)
            acc = Accuracy(pred.cpu().data.numpy(),labels.cpu().data.numpy())
            logger.scalar_summary('Temporal/train_loss',loss.data[0],cnt)
            logger.scalar_summary('Temporal/train_acc',acc,cnt)
            loss.backward()
            optim.step()

            print('epoch: {} cnt: {}'.format(epoch,cnt))

            if cnt%50 == 0:

                imgs,labels = GenVariables(test_dsl)
                pred = model(imgs)
                loss = lossfunc(pred,labels)

                logger.scalar_summary('Temporal/test_loss',loss.data[0],cnt)

                #acc
                acc = Accuracy(pred.cpu().data.numpy(),labels.cpu().data.numpy())
                logger.scalar_summary('Temporal/test_acc',acc,cnt)


        learningrate = learningrate*attenuation
        optim = torch.optim.Adadelta(model.parameters(),lr=learningrate)

        savefile = savepath + 'VGG_Tempora_EX1_{:02d}.pt'.format(epoch%100)
        print('save model to {}'.format(savefile))
        torch.save(module,savefile)


def VGG_Spatial_Net_Run():

    epochs = 10
    loops = 4000
    learningrate = 0.001
    attenuation = 0.1

    train_dsl = train_UCF0101_Spatial()
    test_dsl = test_UCF0101_Spatial()

    model = VGG_Spatial_Net(pretrained=False).cuda()
    lossfunc = nn.CrossEntropyLoss()
    optim = torch.optim.Adadelta(model.parameters(),lr=learningrate)

    cnt = 0

    for epoch in range(epochs) :

        for l in range(loops) :

            cnt+=1

            imgs,labels = GenVariables_Spatial(train_dsl)

            model.zero_grad()
            pred =  model(imgs)
            loss = lossfunc(pred,labels)
            acc = Accuracy(pred.cpu().data.numpy(),labels.cpu().data.numpy())
            logger.scalar_summary('Spatial/train_loss',loss.data[0],cnt)
            logger.scalar_summary('Spatial/train_acc',acc,cnt)
            loss.backward()
            optim.step()

            print('epoch: {} cnt: {}'.format(epoch,cnt))

            if cnt%50 == 0:

                imgs,labels = GenVariables(test_dsl)
                pred = model(imgs)
                loss = lossfunc(pred,labels)
                logger.scalar_summary('Spatial/test_loss',loss.data[0],cnt)

                acc = Accuracy(pred.cpu().data.numpy(),labels.cpu().data.numpy())
                logger.scalar_summary('Spatial/test_acc',acc,cnt)


        learningrate = learningrate*attenuation
        optim = torch.optim.Adadelta(model.parameters(),lr=learningrate)

        savefile = savepath + 'VGG_Spatial_EX1_{:02d}.pt'.format(epoch%100)
        print('save model to {}'.format(savefile))
        torch.save(module,savefile)


if __name__=='__main__':
    VGG_Temporal_Net_Run()
