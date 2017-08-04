import random

import torch
import torch.nn as nn
from torch.autograd import Variable

from VideoClassification.Config import Config as Config
from VideoClassification.model.vgg_twostream.vgg_twostream import VGG_Spatial_Net,VGG_Temporal_Net
from VideoClassification.utils.DataSetLoader.UCF101Loader import test_UCF101_ChooseRandomFromSameVideo,UCF101_TwoStream
from VideoClassification.utils.Logger import Logger


############ Config

logger = Logger(Config.LOGSpace+Config.EX_ID)
savepath = Config.ExWorkSpace+Config.EX_ID+'/'

import os.path
if os.path.isdir(savepath)==False:
    os.mkdir(savepath)

batchsize = 10

############

spa_model_save_file = None
tem_model_save_file = None

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

    dsl = test_UCF101_ChooseRandomFromSameVideo(dsl=UCF101_TwoStream)

    for i in range(loops):
        imgpaths,labels = random.choice(dsl)
        

if __name__=='__main__':
    pass
