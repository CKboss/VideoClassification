import random
import threading
from queue import Queue

import numpy as np
import torch
from torch.autograd import Variable

import VideoClassification.Config.Config as Config
from VideoClassification.utils.DataSetLoader.UCF101_DataSetLoader_FromFileName.UCF101Loader import test_UCF101_C3D,test_UCF0101_Spatial,test_UCF0101_Temporal
from VideoClassification.utils.DataPretreatment.PipeLine import GenTensors, ImgAugPipes
from VideoClassification.utils.DataSetLoader.UCF101_DataSetLoader_MYSQL.UCF101_DBtools import getFrames_imgfilepath,getTemporals_imgfilepath

try:
    from cv2 import cv2
except:
    import cv2

def GenVariables_C3D(splitkind='test',batchsize=8,**kwargs):

    imgpathss = []
    labels = []

    while len(labels) < batchsize:
        items = getFrames_imgfilepath(splitkind)
        if len(items)<20:
            continue
        else :
            imgpathss.append([ Config.UCF101_images_root+item[0] for item in items[:20]])
            labels.append(items[0][1])

    imgs = GenTensors(imgpathss,isTemporal=False,outputshape=(112,112),isNormal=False)
    imgs = torch.transpose(imgs,1,2)

    # TODO check bug

    imgs = Variable(imgs,**kwargs).float()
    labels = Variable(torch.from_numpy(np.array(labels)),**kwargs).long()

    return imgs,labels


def GenVariables_Temporal(splitkind='test',batchsize=8,**kwargs):


    imgpathss = []
    labels = []

    while len(labels) < batchsize:
        items = getTemporals_imgfilepath(splitkind)
        if len(items)<20:
            continue
        else :
            imgpathss.append([ Config.UCF101_images_root+item[0] for item in items[:20]])
            labels.append(items[0][1])

    # TODO add requires_grad params

    imgs = GenTensors(imgpathss,isTemporal=True)

    # imgs.size() => (8x20x224x224)

    imgs = Variable(imgs,**kwargs).float()
    labels = Variable(torch.from_numpy(np.array(labels)),**kwargs).long()

    return imgs,labels

def GenVariables_Spatial(splitkind='test',batchsize=8,**kwargs):

    # TODO add requires_grad params

    imgpaths = []
    labels = []

    while len(labels) < batchsize:
        items = getFrames_imgfilepath(splitkind)
        if len(items)<1:
            continue
        else :
            imgpaths.append([ Config.UCF101_images_root+items[0][0]])
            labels.append(items[0][1])

    imgs = []

    for path in imgpaths:
        imgs.append(cv2.imread(path[0]))

    imgs = np.array(ImgAugPipes(imgs))
    # imgs.size() => (8x3x224x224)

    imgs = Variable(torch.from_numpy(imgs),**kwargs).float()
    labels = Variable(torch.from_numpy(np.array(labels)),**kwargs).long()

    return imgs,labels

def GenVariables_VideoSpatialAndTemporal(dsl,batchsize):
    # TODO Need To Change
    raise NotImplementedError



class PictureQueue(object):

    def __init__(self,dsl,Gen,batchsize=8,worker=20,mxsize=32):
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
    dsl = test_UCF0101_Spatial()
    dsl = test_UCF0101_Temporal()

    itemss = random.choices(dsl,k=3)

    pq = PictureQueue(dsl,GenVariables_C3D,3)

    pa,pb = pq.Get()
