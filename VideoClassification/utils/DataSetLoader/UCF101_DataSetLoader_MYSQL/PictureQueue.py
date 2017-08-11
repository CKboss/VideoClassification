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

    # TODO add requires_grad params

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
    dsl = test_UCF0101_Spatial()
    dsl = test_UCF0101_Temporal()

    itemss = random.choices(dsl,k=3)

    pq = PictureQueue(dsl,GenVariables_C3D,3)

    pa,pb = pq.Get()
