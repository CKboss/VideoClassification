import os
import random
from torch.utils.data import Dataset,DataLoader
import numpy as np

import VideoClassification.Config.Config as Config


# Config.UCF101_images_root = '/datacenter/1/UCF_Data/'

image_num = dict()
image_id = dict()

def GaoImageID():
    if len(image_id) == 0:
        with open(Config.Code_root+'/data/classInd.txt','r') as f:
            for line in f.readlines():
                value,key = line.split(' ')
                image_id[key[:-1]] = int(value)-1


class UCF101_Spatial(Dataset):

    def __init__(self,file):
        with open(file,'r') as f:
            self.items = [ line[:-1] for line in f.readlines()]
            GaoImageID()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, id):

        filepath = self.items[id].split('.')[0]

        classname = filepath.split('/')[0]
        classid = image_id[classname]

        filepath = Config.UCF101_images_root + filepath + '/image/'

        ####
        # filepath = '/home/lab/Desktop/Development/dense_flow_fbf/testfile-fbf/UCF101_images/ApplyLipstick/v_ApplyLipstick_g01_c02/image/'

        n = image_num.get(filepath,-1)
        if n==-1:
            n = image_num[filepath] = len(os.listdir(filepath))

        n = random.randint(1,n)

        filepath = filepath+'image_{:04d}.jpg'.format(n)

        return filepath,classid

class UCF101_Temporal(Dataset):

    def __init__(self,file):
        with open(file,'r') as f:
            self.items = [ line[:-1] for line in f.readlines()]
            GaoImageID()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, id):

        filepath = self.items[id].split('.')[0]

        classname = filepath.split('/')[0]
        classid = image_id[classname]

        filepath_x = Config.UCF101_images_root + filepath + '/flow_y/'
        filepath_y = Config.UCF101_images_root + filepath + '/flow_x/'

        ####
        # filepath_x = '/home/lab/Desktop/Development/dense_flow_fbf/testfile-fbf/UCF101_images/ApplyLipstick/v_ApplyLipstick_g01_c02/flow_x/'
        # filepath_y = '/home/lab/Desktop/Development/dense_flow_fbf/testfile-fbf/UCF101_images/ApplyLipstick/v_ApplyLipstick_g01_c02/flow_y/'

        n = image_num.get(filepath,-1)
        if n==-1:
            n = image_num[filepath] = len(os.listdir(filepath_x))

        i = random.randint(1,n)

        l = max(i-4,1)
        r = min(n,i+5)

        if l==1 : r = 10
        if r==n : l = n-9

        filepathlist = []

        for i in range(l,r+1):
            filepathlist.append(filepath_x+'flow_x_{:04d}.jpg'.format(i))
            filepathlist.append(filepath_y+'flow_y_{:04d}.jpg'.format(i))

        return filepathlist,classid

class Resize(object):
    pass


def test_UCF0101_Spatial():
    return UCF101_Spatial(Config.Code_root+'/data/testlist01.txt')

def train_UCF0101_Spatial():
    return UCF101_Spatial(Config.Code_root+'/data/trainlist01.txt')

def test_UCF0101_Temporal():
    return UCF101_Temporal(Config.Code_root+'/data/testlist01.txt')

def train_UCF0101_Temporal():
    return UCF101_Temporal(Config.Code_root+'/data/trainlist01.txt')

if __name__=='__main__':

    # TODO
    # DataLoad 连续取多张图片有问题 !!!!


    testloader = [ DataLoader(test_UCF0101_Spatial(),drop_last=True),
                   DataLoader(test_UCF0101_Temporal(),drop_last=True),
                   DataLoader(train_UCF0101_Spatial(),drop_last=True),
                   DataLoader(train_UCF0101_Temporal(),drop_last=True)]

    testloader = DataLoader(test_UCF0101_Temporal(),drop_last=True,batch_size=3)

    tt = test_UCF0101_Temporal()

    import random
    random.choices(tt,k=3)


    for i,item in enumerate(testloader):
        print(item)

    imgpaths = []
    for tl in testloader :
        for i,items in enumerate(tl):
            if len(items) > 0:
                for item in items:
                    print(item[0])
                    imgpaths.append(item[0])
                break
        break

    from VideoClassification.utils.data_pretreatment.PipeLine import ImgAugPipes,Imread_lru
    import matplotlib.pyplot as plt


    imgs = []
    for imgpath in imgpaths :
        imgs.append(Imread_lru(imgpath))

    for img in imgs :
        print(img.shape)

    aimgs = ImgAugPipes(imgs)

    print(aimgs.shape)


    # len(ds)
    # ds.items[-1]
    # ds[3][0]
    #
    # len(testloader)
    #
    # for i,item in enumerate(testloader):
    #     print(i,item)
    #
    # GaoImageID()
    #
    # random.randint(0,10)
    #
    # with open('./data/classInd.txt','r') as f:
    #     for line in f.readlines():
    #         print(line)
