import random

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

spa_model_save_file = Config.Server_Root_Path+'pretrainedmodel/Sptail.pt'
tem_model_save_file = Config.Server_Root_Path+'pretrainedmodel/Temporal.pt'

dsl = test_UCF101_ChooseOrderFromSameVideo(dsl=UCF101_TwoStream)
def gen():
    return GenVariables_VideoSpatialAndTemporal(dsl=dsl,batchsize=4)

def VGG_TwoStream_Video_AVG_Merge_Test():
    '''
    Video Level VGG TwoStream AVG merge evalution
    '''
    loops = 10

    spa_model = VGG_Spatial_Net().cuda()
    tem_model = VGG_Temporal_Net().cuda()

    if spa_model_save_file is not None:
        spa_model.load_state_dict(torch.load(spa_model_save_file))
    if tem_model_save_file is not None:
        tem_model.load_state_dict(torch.load(tem_model_save_file))

    imgs,labels = gen()

    labels = Variable(torch.from_numpy(labels)).cuda().long()

    n,b,c,w,h = imgs.shape

    for l in range(loops):

        for i in range(n):

            spatial_input = Variable(torch.from_numpy(imgs[i,:,0:3,:,:])).cuda().float()
            temporal_input = Variable(torch.from_numpy(imgs[i,:,3:,:,:])).cuda().float()

            predict_1 = spa_model.inference(spatial_input)
            predict_2 = tem_model.inference(temporal_input)

            predict_all = (predict_1+predict_2)/2

            predict_all = torch.cumsum(predict_all,0)
            predict_all = predict_all[-1,:]
            predict_all = predict_all / b

            acc = accuracy(predict_all,labels[i],topk=(1,5,10))

            print('nth: {} -----> '.format(n))
            print('acc@1: ',acc[0])
            print('acc@5: ',acc[1])
            print('acc@10: ',acc[2])

def VGG_Two_Stream_TSN():
    pass

if __name__=='__main__':
    pass
