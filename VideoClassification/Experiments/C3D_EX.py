import torch
import torch.nn as nn
from torch.autograd import Variable

try:
    from cv2 import cv2
except:
    import cv2

import numpy as np


import VideoClassification.Config.Config as Config
from VideoClassification.model.C3D.C3D import C3D
from VideoClassification.utils.Logger import Logger
from VideoClassification.utils.DataSetLoader.UCF101Loader import test_UCF101_C3D,train_UCF101_C3D
from VideoClassification.utils.toolkits import accuracy,try_to_load_state_dict
from VideoClassification.utils.DataSetLoader.PictureQueue import PictureQueue,GenVariables_C3D


'''
C3D
'''


############ Config

logger = Logger(Config.LOGSpace+Config.EX_ID)
savepath = Config.ExWorkSpace+Config.EX_ID+'/'

import os.path
if os.path.isdir(savepath)==False:
    os.mkdir(savepath)

batchsize = 30

############


def C3D_Net_Run():

    epochs = 80
    loops = 2000
    learningrate = 0.1
    attenuation = 0.5

    model = C3D().cuda()

    if Config.LOAD_SAVED_MODE_PATH is not None :
        import types
        model.try_to_load_state_dict = types.MethodType(try_to_load_state_dict,model)
        model.try_to_load_state_dict(torch.load(Config.LOAD_SAVED_MODE_PATH))
        print('LOAD {} done!'.format(Config.LOAD_SAVED_MODE_PATH))

    lossfunc = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(),lr=learningrate)

    pq_train = PictureQueue(dsl=train_UCF101_C3D(),Gen=GenVariables_C3D,batchsize=batchsize)
    pq_test = PictureQueue(dsl=test_UCF101_C3D(),Gen=GenVariables_C3D,batchsize=batchsize)

    cnt = 0
    for epoch in range(epochs) :

        for l in range(loops) :

            cnt+=1

            # print('--------------')
            imgs,labels = pq_train.Get()
            # print('imgs size: ',imgs.size())

            model.zero_grad()

            # print('{} before pred'.format(cnt))
            pred =  model(imgs)
            # print('{} before loss'.format(cnt))

            loss = lossfunc(pred,labels)

            logger.scalar_summary('C3D/train_loss',loss.data[0],cnt)

            # print('{} before bp'.format(cnt))
            loss.backward()

            # print('{} before optim'.format(cnt))
            optim.step()


            # print('C3D epoch: {} cnt: {} loss: {}'.format(epoch,cnt,loss.data[0]))

            if cnt%20 == 0:

                imgs,labels = pq_test.Get()
                pred = model(imgs)
                loss = lossfunc(pred,labels)

                logger.scalar_summary('C3D/test_loss',loss.data[0],cnt)

                #acc
                acc = accuracy(pred,labels,topk=(1,5,10))
                logger.scalar_summary('C3D/test_acc@1',acc[0],cnt)
                logger.scalar_summary('C3D/test_acc@5',acc[1],cnt)
                logger.scalar_summary('C3D/test_acc@10',acc[2],cnt)


                imgs,labels = pq_train.Get()
                pred = model(imgs)

                acc = accuracy(pred,labels,topk=(1,5,10))
                logger.scalar_summary('C3D/train_acc@1',acc[0],cnt)
                logger.scalar_summary('C3D/train_acc@5',acc[1],cnt)
                logger.scalar_summary('C3D/train_acc@10',acc[2],cnt)

            if cnt%2000 == 0:
                savefile = savepath + 'C3D_EX1_{:02d}.pt'.format(epoch%20)
                print('C3D save model to {}'.format(savefile))
                torch.save(model,savefile)

        if epoch in [20,40,60]:
            learningrate = learningrate*attenuation
            optim = torch.optim.Adam(model.parameters(),lr=learningrate)


if __name__=='__main__':
    # C3D_Net_Run()
    pass
