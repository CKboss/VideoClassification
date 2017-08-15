
try:
    from cv2 import cv2
except:
    import cv2

import random
import numpy as np
import torch
import torch.nn.functional as F
from VideoClassification.utils.DataSetLoader.UCF101_DataSetLoader_FromFileName.UCF101Loader import test_UCF101_ChooseOrderFromSameVideo, UCF101_TwoStream,UCF101_Spatial,test_UCF0101_Spatial,test_UCF0101_Temporal,test_UCF101_ChooseRandomFromSameVideo
from VideoClassification.utils.Others.Logger import Logger
from torch.autograd import Variable

from VideoClassification.Config import Config as Config
from VideoClassification.model.resnet_twostream.resnet_twostream import resnet152_SpatialNet,resnet50_TemporalNet
from VideoClassification.utils.DataSetLoader.UCF101_DataSetLoader_FromFileName.PictureQueue import \
    GenVariables_VideoSpatialAndTemporal,GenVariables_Spatial,GenVariables_Temporal,GenTensors
from VideoClassification.utils.Others.toolkits import accuracy
from VideoClassification.utils.DataPretreatment.PipeLine import ImgAugPipes

############ Config

logger = Logger(Config.LOGSpace+Config.EX_ID)
savepath = Config.ExWorkSpace+Config.EX_ID+'/'

import os.path
if os.path.isdir(savepath)==False:
    os.mkdir(savepath)

batchsize = 10

############

spa_model_save_file = Config.Server_Root_Path+'pretrainedmodel/ResNet152_Spatial.pt'
tem_model_save_file = Config.Server_Root_Path+'pretrainedmodel/ResNet50_Temporal.pt'


def TestOnlySpatialNet():

    spa_model = resnet152_SpatialNet().cuda().eval()
    if spa_model_save_file is not None:
        spa_model.load_state_dict(torch.load(spa_model_save_file))
        print('load spa_model success!')

    dsl = test_UCF0101_Spatial()
    def gen():
        return GenVariables_Spatial(dsl,batchsize=16)

    loops = 50
    correct_1 = 0
    correct_5 = 0
    correct_10 = 0

    for i in range(loops):
        images,labels = gen()
        pred = spa_model(images)

        # pred = F.softmax(pred)

        labels = labels.cpu()
        pred = pred.cpu()

        acc = accuracy(pred,labels,topk=(1,5,10))

        correct_1 += acc[0]
        correct_5 += acc[1]
        correct_10 += acc[2]

        print(acc)

    print('acc@1:',correct_1/loops)
    print('acc@5:',correct_5/loops)
    print('acc@10:',correct_10/loops)

    '''
    
    resnet151_Spatial without Softmax:
    acc@1:  69.5625
    acc@5:  87.8125
    acc@10:  92.9375
    
    
    resnet151_Spatial Softmax:
    acc@1:  68.625
    acc@5:  88.875
    acc@10:  94.0625
    
    
    '''

def TestOnlyTemporalNet():

    tem_model = resnet50_TemporalNet().cuda()
    if tem_model_save_file is not None:
        tem_model.load_state_dict(torch.load(tem_model_save_file))
        print('load tem_model success!')

    # tem_model.eval()

    dsl = test_UCF0101_Temporal()
    def gen():
        return GenVariables_Temporal(dsl,batchsize=16)

    loops = 100
    correct_1 = 0
    correct_5 = 0
    correct_10 = 0

    for l in range(loops):
        images,labels = gen()
        pred = tem_model(images)

        # pred = F.softmax(pred)

        labels = labels.cpu()
        pred = pred.cpu()

        acc = accuracy(pred,labels,topk=(1,5,10))

        correct_1 += acc[0]
        correct_5 += acc[1]
        correct_10 += acc[2]

        print(l,':',acc)

    print('acc@1: ',correct_1/loops)
    print('acc@5: ',correct_5/loops)
    print('acc@10: ',correct_10/loops)

    '''
    acc@1:  38.625
    acc@5:  67.375
    acc@10:  78.375
    
    acc@1:  42.708333333333336
    acc@5:  68.125
    acc@10:  78.33333333333333
    
    acc@1:  41.25
    acc@5:  66.6875
    acc@10:  77.875
    '''

def Only_Merge_Spatial_Net_Test():


    spa_model = resnet152_SpatialNet().cuda().eval()
    if spa_model_save_file is not None:
        spa_model.load_state_dict(torch.load(spa_model_save_file))
        print('load spa_model success!')

    dsl = test_UCF101_ChooseRandomFromSameVideo(dsl=UCF101_Spatial)

    # dsl = test_UCF0101_Spatial()
    # def gen():
    #     return GenVariables_Spatial(dsl,batchsize=16)

    loops = 500
    correct_1 = 0
    correct_5 = 0
    correct_10 = 0

    for l in range(loops):
        imgpaths,Lables = random.choice(dsl)

        imgs = []
        for path in imgpaths:
            imgs.append(cv2.imread(path))
        imgs = np.array(imgs)
        imgs = ImgAugPipes(imgs,NoAug=True,isNormal=True)

        imgs = Variable(torch.from_numpy(imgs)).cuda().float()
        lable = Variable(torch.from_numpy(np.array(Lables))).cuda().long()

        pred = spa_model(imgs[:3])

        lable = Variable(torch.from_numpy(np.array([Lables[0]]))).cuda().long()
        pred = pred.sum(0)/3

        # acc = accuracy(pred.cpu(),lable.cpu(),(1,5,10))
        acc = accuracy(pred.cpu(),lable.cpu(),(1,5,10))


        correct_1 += acc[0]
        correct_5 += acc[1]
        correct_10 += acc[2]

        print(l,':',acc)

    print('acc@1:',correct_1/loops)
    print('acc@5:',correct_5/loops)
    print('acc@10:',correct_10/loops)

    '''
    acc@1: 72.25
    acc@5: 91.0
    acc@10: 95.625
    '''

def Only_Merge_Temporal_Net():

    dsl = test_UCF101_ChooseRandomFromSameVideo(dsl=UCF101_TwoStream)

    tem_model = resnet50_TemporalNet().cuda()
    if tem_model_save_file is not None:
        tem_model.load_state_dict(torch.load(tem_model_save_file))
        print('load tem_model success!')

    tem_model.eval()

    loops = 20
    correct_1 = 0
    correct_5 = 0
    correct_10 = 0

    for l in range(loops):
        imgpathss,labels = random.choice(dsl)
        imgpathss = [imgpaths[1:] for imgpaths in imgpathss]
        imgs = GenTensors(imgpathss,isTemporal=True)

        imgs = Variable(imgs).cuda()
        labels = Variable(torch.from_numpy(np.array(labels))).cuda().long()

        pred = tem_model(imgs)

        # acc = accuracy(pred,labels,topk=(1,5,10))

        pred = pred.sum(0)/8
        lable = labels[0].cuda().long()

        acc = accuracy(pred,lable,topk=(1,5,10))

        print(l,':',acc)

        correct_1 += acc[0]
        correct_5 += acc[1]
        correct_10 += acc[2]


    print('acc@1:',correct_1/loops)
    print('acc@5:',correct_5/loops)
    print('acc@10:',correct_10/loops)

def Merge_Test_2():

    dsl = test_UCF101_ChooseRandomFromSameVideo(dsl=UCF101_TwoStream)

    def gen():
        return GenVariables_VideoSpatialAndTemporal(dsl=dsl,batchsize=4)

    spa_model = resnet152_SpatialNet().cuda()
    tem_model = resnet50_TemporalNet().cuda()


    if spa_model_save_file is not None:
        spa_model.load_state_dict(torch.load(spa_model_save_file))
        print('load spa_model success!')
    if tem_model_save_file is not None:
        tem_model.load_state_dict(torch.load(tem_model_save_file))
        print('load tem_model success!')

    spa_model.eval()
    tem_model.eval()

    dsl = test_UCF101_ChooseRandomFromSameVideo(dsl=UCF101_TwoStream)

    loops = 20
    correct_1 = 0
    correct_5 = 0
    correct_10 = 0

    for l in range(loops):
        imgpathss,labels = random.choice(dsl)
        imgpaths_1 = [imgpaths[0] for imgpaths in imgpathss]
        imgpaths_2 = [imgpaths[1:] for imgpaths in imgpathss]
        imgs = GenTensors(imgpaths_2,isTemporal=True)

        imgs = Variable(imgs).cuda()
        labels = Variable(torch.from_numpy(np.array(labels))).cuda().long()

        pred = tem_model(imgs)

        # acc = accuracy(pred,labels,topk=(1,5,10))

        pred = pred.sum(0)/8
        lable = labels[0].cuda().long()

        acc = accuracy(pred,lable,topk=(1,5,10))

        print(l,':',acc)

        correct_1 += acc[0]
        correct_5 += acc[1]
        correct_10 += acc[2]


    print('acc@1:',correct_1/loops)
    print('acc@5:',correct_5/loops)
    print('acc@10:',correct_10/loops)


def Merge_Test():
    '''
    Video Level VGG TwoStream AVG merge evalution
    '''

    dsl = test_UCF101_ChooseRandomFromSameVideo(dsl=UCF101_TwoStream)

    def gen():
        return GenVariables_VideoSpatialAndTemporal(dsl=dsl,batchsize=4)

    spa_model = resnet152_SpatialNet().cuda()
    tem_model = resnet50_TemporalNet().cuda()


    if spa_model_save_file is not None:
        spa_model.load_state_dict(torch.load(spa_model_save_file))
        print('load spa_model success!')
    if tem_model_save_file is not None:
        tem_model.load_state_dict(torch.load(tem_model_save_file))
        print('load tem_model success!')

    spa_model.eval()
    # tem_model.eval()

    imgs,labels = gen()

    labels = Variable(torch.from_numpy(labels)).cuda().long()

    n,b,c,w,h = imgs.shape

    correct_d = [0,0,0]
    correct_spa = [0,0,0]
    correct_tmp = [0,0,0]


    loops = 30
    for l in range(loops):

        imgs,labels = gen()

        labels = Variable(torch.from_numpy(labels)).cuda().long()

        print('\n')
        print('loop:',l)
        print('test num: ',l*n)
        print('correct_d',correct_d)
        print('correct_spa',correct_spa)
        print('correct_tmp',correct_tmp)

        for i in range(n):

            label = labels[i][0]

            spatial_input = Variable(torch.from_numpy(imgs[i,:,0:3,:,:])).cuda().float()
            temporal_input = Variable(torch.from_numpy(imgs[i,:,3:,:,:])).cuda().float()

            pred_spa = spa_model.inference(spatial_input)
            pred_tem = tem_model.inference(temporal_input)

            # pred_spa = F.softmax(pred_spa)
            # pred_tem = F.softmax(pred_tem)

            t1_spa = pred_spa.sum(0)/b
            t1_tem = pred_tem.sum(0)/b

            acc_spa = accuracy(t1_spa,label,topk=(1,5,10))
            for j in range(3):
                if acc_spa[j] > 0:
                    correct_spa[j]+=1

            acc_tmp = accuracy(t1_tem,label,topk=(1,5,10))
            for j in range(3):
                if acc_tmp[j] > 0:
                    correct_tmp[j]+=1

            pred_merge = F.softmax((pred_spa.sum(0)/b+pred_tem.sum(0)/b)/ 2)

            acc_merge = accuracy(pred_merge,label,topk=(1,5,10))
            for j in range(3):
                if acc_merge[j] > 0:
                    correct_d[j]+=1



if __name__=='__main__':
    pass
