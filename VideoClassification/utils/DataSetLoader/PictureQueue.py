import random
import threading
from queue import Queue

import numpy as np

import torch
from torch.autograd import Variable

from VideoClassification.utils.data_pretreatment.PipeLine import GenTensors,ImgAugPipes
from VideoClassification.utils.DataSetLoader.UCF101Loader import test_UCF0101_Spatial,test_UCF0101_Temporal,test_UCF101_C3D

try:
    from cv2 import cv2
except:
    import cv2

def GenVariables_C3D(dsl,batchsize=8,**kwargs):

    # TODO add requires_grad params
    items = random.choices(dsl,k=batchsize)

    imgpathss = []
    labels = []

    for item in items:
        imgpathss.append(item[0])
        labels.append(item[1])

    imgs = GenTensors(imgpathss,isTemporal=False,outputshape=(112,112))

    imgs = torch.transpose(imgs,1,2)

    imgs = Variable(imgs,**kwargs).float()
    labels = Variable(torch.from_numpy(np.array(labels)),**kwargs).long()

    return imgs,labels


def GenVariables_Temporal(dsl,batchsize=8,**kwargs):

    # TODO add requires_grad params
    items = random.choices(dsl,k=batchsize)

    imgpathss = []
    labels = []

    for item in items:
        imgpathss.append(item[0])
        labels.append(item[1])

    imgs = GenTensors(imgpathss,isTemporal=True)

    imgs = Variable(imgs,**kwargs).float()
    labels = Variable(torch.from_numpy(np.array(labels)),**kwargs).long()

    return imgs,labels

def GenVariables_Spatial(dsl,batchsize=8,**kwargs):

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
    imgs = Variable(torch.from_numpy(imgs),**kwargs).float()
    labels = Variable(torch.from_numpy(np.array(labels)),**kwargs).long()

    return imgs,labels

class PictureQueue(object):

    def __init__(self,dsl,Gen,batchsize=8,worker=10,mxsize=32):
        self.dsl = dsl
        self.Gen = Gen
        self.worker = worker
        self.q = Queue(mxsize)
        self.batchsize = batchsize
        self.ts = []
        for i in range(worker):
            self.ts.append(threading.Thread(target=self.pr,name='Producter_{}'.format(i)))
        for i in range(worker):
            self.ts[i].start()

    def pr(self):
        while True:
            self.q.put(self.Gen(self.dsl,self.batchsize))

    def Get(self):
        imgs,labels = self.q.get()
        return imgs.cuda(),labels.cuda()

    def Close(self):
        '''
        May Be Need To Stop Threads
        :return:
        '''
        pass

if __name__=='__main__':

    dsl = test_UCF101_C3D()

    pq = PictureQueue(dsl,GenVariables_C3D,3)

    pa,pb = pq.Get()
