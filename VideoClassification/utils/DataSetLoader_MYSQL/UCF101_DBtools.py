import os
import os.path
import sys

import VideoClassification.Config.Config as Config
from VideoClassification.utils.DataSetLoader.UCF101Loader import GaoImageID,image_id
from VideoClassification.utils.DBTools.MySql.dbSQL import INSERT_NEW_IMAGE

testlst = []

def getTrain_dict():
    if len(testlst) == 0:
        with open(Config.Code_root+'/data/testlist01.txt','r') as f:
            for line in f.readlines():
                videoname = line.split(' ')[0]
                videoname = videoname[:-5]
                testlst.append(videoname.split('/')[1])


line = '/datacenter/1/UCF_Data/UCF101_images/FrisbeeCatch/v_FrisbeeCatch_g01_c01/flow_x'

def chuli(line):

    items = line.split('/')
    classname = items[-3]
    label = image_id[classname]
    videoname = items[-2]

    if videoname in testlst:
        splitkind = 'test'
    else:
        splitkind = 'train'

    if items[-1][:4] == 'flow':
        imgkind = 'flow'
    else :
        imgkind = 'frame'

    pathprefix = '/'.join(items[5:])

    return splitkind,imgkind,videoname,label,pathprefix


chuli(line)


# , splitkind [train/test/val] , imgfilepath [] , imgfilename [] \
#     imgkind [frame/optial] , video_name , label

def FindAndInsertImages():

    GaoImageID()
    getTrain_dict()

    for root,dirs,files in os.walk(Config.UCF101_images_root):
        if len(files) > 0:
            print('--> ',root)
            need_to_insert = []
            for file in files:
                splitkind,imgkind,videoname,label,prefixpath = chuli(root)
                imgname = file
                imgpath = prefixpath + '/' + file
                need_to_insert.append([splitkind,imgpath,imgname,imgkind,videoname,label])
            InsertInToImages(need_to_insert)

