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


############ Config

logger = Logger(Config.ExWorkSpace+'/LOG/EX1')
savepath = Config.ExWorkSpace+'/EX1/'
epochs = 20
loops = 1000
learningrate = 0.01
attenuation = 0.5
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
            loss.backward()
            optim.step()


            if cnt%50 == 0:

                imgs,labels = GenVariables(train_dsl)
                loss = lossfunc(pred,labels)
                logger.scalar_summary('loss',loss.data[0],cnt)


        savefile = savepath + 'VGG_Tempora_{:02d}.pt'.format(epoch%100)
        torch.save(module,savefile)


def VGG_Spatial_Net_Run():

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
            loss.backward()
            optim.step()


            if cnt%50 == 0:

                imgs,labels = GenVariables(train_dsl)
                loss = lossfunc(pred,labels)
                logger.scalar_summary('loss',loss.data[0],cnt)


        savefile = savepath + 'VGG_Tempora_{:02d}.pt'.format(epoch%100)
        torch.save(module,savefile)


if __name__=='__main__':
    cv2.imread()
