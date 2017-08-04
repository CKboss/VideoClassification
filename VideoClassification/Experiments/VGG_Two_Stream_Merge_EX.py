import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from VideoClassification.Config import Config as Config
from VideoClassification.model.vgg_twostream.vgg_twostream import VGG_Spatial_Net,VGG_Temporal_Net
from VideoClassification.utils.DataSetLoader.UCF101Loader import test_UCF101_ChooseOrderFromSameVideo,UCF101_TwoStream
from VideoClassification.utils.DataSetLoader.PictureQueue import GenVariables_VideoSpatialAndTemporal
from VideoClassification.utils.toolkits import accuracy
from VideoClassification.utils.Logger import Logger


############ Config

logger = Logger(Config.LOGSpace+Config.EX_ID)
savepath = Config.ExWorkSpace+Config.EX_ID+'/'

import os.path
if os.path.isdir(savepath)==False:
    os.mkdir(savepath)

batchsize = 10

############

spa_model_save_file = Config.Server_Root_Path+'pretrainedmodel/Spatial.pt'
tem_model_save_file = Config.Server_Root_Path+'pretrainedmodel/Temporal.pt'

dsl = test_UCF101_ChooseOrderFromSameVideo(dsl=UCF101_TwoStream)
def gen():
    return GenVariables_VideoSpatialAndTemporal(dsl=dsl,batchsize=4)

def VGG_TwoStream_Video_AVG_Merge_Test():
    '''
    Video Level VGG TwoStream AVG merge evalution
    '''
    loops = 100

    spa_model = VGG_Spatial_Net().cuda()
    tem_model = VGG_Temporal_Net().cuda()

    if spa_model_save_file is not None:
        spa_model.load_state_dict(torch.load(spa_model_save_file))
        print('load spa_model success!')
    if tem_model_save_file is not None:
        tem_model.load_state_dict(torch.load(tem_model_save_file))
        print('load tem_model success!')

    imgs,labels = gen()

    labels = Variable(torch.from_numpy(labels)).cuda().long()

    n,b,c,w,h = imgs.shape

    correct_d = [0,0,0]
    correct_spa = [0,0,0]
    correct_tmp = [0,0,0]

    for l in range(loops):

        imgs,labels = gen()

        print('l: ',l)
        print('correct_d',correct_d)
        print('correct_spa',correct_spa)
        print('correct_tmp',correct_tmp)

        for i in range(n):

            spatial_input = Variable(torch.from_numpy(imgs[i,:,0:3,:,:])).cuda().float()
            temporal_input = Variable(torch.from_numpy(imgs[i,:,3:,:,:])).cuda().float()

            # print(labels)

            # print('spatial_input: ',spatial_input.size())
            # print('temporal_input: ',temporal_input.size())

            predict_1 = spa_model.inference(spatial_input)
            predict_2 = tem_model.inference(temporal_input)

            # print('predict_1: ',predict_1.size())
            # print('predict_2: ',predict_2.size())

            predict_all = (predict_1+predict_2)/2

            predict_all = torch.cumsum(predict_all,0)
            predict_all = predict_all[-1,:]
            predict_all = predict_all / b
            predict_all = predict_all.view(1,101)

            target = np.array([labels[i,0]])
            target = torch.from_numpy(target).view(1,1).cuda().long()
            target = Variable(target)
            # print('target: ',target)

            # print('predict_all: ',predict_all.size())

            predict_1 = torch.cumsum(predict_1,0)
            predict_1 = predict_1[-1].view(1,101)
            predict_1 = predict_1 / b
            acc = accuracy(predict_1,target,topk=(1,5,10))

            for ii in range(3):
                if acc[ii] > 0.5:
                    correct_spa[ii] += 1

            # print(' only avg spatial net acc:')
            # print('acc@1: ',acc[0])
            # print('acc@5: ',acc[1])
            # print('acc@10: ',acc[2])
            # print('-'*20)
            #
            #
            predict_2 = torch.cumsum(predict_2,0)
            predict_2 = predict_2[-1].view(1,101)
            predict_2 = predict_2 / b
            acc = accuracy(predict_2,target,topk=(1,5,10))
            # print(' only avg temporal net acc:')
            # print('acc@1: ',acc[0])
            # print('acc@5: ',acc[1])
            # print('acc@10: ',acc[2])
            # print('-'*20)


            for ii in range(3):
                if acc[ii] > 0.5:
                    correct_tmp[ii] += 1

            acc = accuracy(predict_all,target,topk=(1,5,10))
            # print(' avg merge two net acc: ')
            # print('acc@1: ',acc[0])
            # print('acc@5: ',acc[1])
            # print('acc@10: ',acc[2])
            # print('-'*20)

            for ii in range(3):
                if acc[ii] > 0.5:
                    correct_d[ii] += 1

def VGG_Two_Stream_TSN():
    pass

if __name__=='__main__':
    pass
