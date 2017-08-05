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

def GenVariables_VideoSpatialAndTemporal(dsl,batchsize):

    itemss = random.choices(dsl,k=batchsize)

    ret_imgs= []
    ret_labels = []

    for b in range(batchsize):

        # 8x21 picture paths and 8 same labels
        imgpathss,labels = itemss[b]

        n = len(labels)
        m = len(imgpathss[0])

        tmp_ret_imgs= []
        tmp_ret_labels = []

        for i in range(n):
            # every batch has 8~10 pictures

            imgpaths = imgpathss[i]
            # first img is origin image
            origin_img = cv2.imread(imgpaths[0])

            # the other is temporal image
            temporal_imgs = []
            for j in range(1,m):
                temporal_imgs.append(cv2.imread(imgpaths[j]))

            # OK now concate them
            imgs = [origin_img] + temporal_imgs
            imgs = np.array(imgs)

            # use img Aug on them
            # imgs.shape is (21,3,224,224)

            imgs = ImgAugPipes(imgs)

            # now just change it to the tensor and add to ret_imgs

            temp_array = imgs[0,:,:,:]

            for j in range(1,m):
                t = imgs[j,0,:,:]
                t = np.reshape(t,(1,224,224))
                temp_array = np.vstack((temp_array,t))

            tmp_ret_imgs.append(temp_array)
            tmp_ret_labels.append(labels[0])

        ret_imgs.append(tmp_ret_imgs)
        ret_labels.append(tmp_ret_labels)

    ret_imgs = np.array(ret_imgs)
    ret_labels = np.array(ret_labels)

    return ret_imgs,ret_labels


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

    pq = PictureQueue(dsl,GenVariables_C3D,3)

    pa,pb = pq.Get()
