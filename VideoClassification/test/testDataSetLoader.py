import VideoClassification.Config.Config as Config
from VideoClassification.utils.DataSetLoader.UCF101_DataSetLoader_FromFileName.UCF101Loader import \
    ChooseRandomFromSameVideo, UCF101_TwoStream, UCF101_C3D, ChooseOrderFromSameVideo

cfsv = ChooseOrderFromSameVideo(file=Config.Code_root+'/data/testlist01.txt', dsl=UCF101_TwoStream)

cfsv2 = ChooseRandomFromSameVideo(file=Config.Code_root+'/data/testlist01.txt', dsl=UCF101_TwoStream)

cfsv3 = ChooseRandomFromSameVideo(file=Config.Code_root+'/data/testlist01.txt', dsl=UCF101_C3D)


a,b = cfsv2[0]

filelists,lbs = cfsv2[1000]

lsts = list(zip(filelists,lbs))

if True:
    first_img = filelists[0]
    lsts = lsts[1:]

filelists = []

tims = [ int(files[0][0][-8:-4]) for files in lsts ]

ll = sorted(list(zip(tims,filelists)))

[l[1] for l in ll]

lbs

##########################################################################################

import random

import numpy as np

from VideoClassification.utils.data_pretreatment.PipeLine import ImgAugPipes

import matplotlib.pyplot as plt

try:
    from cv2 import cv2
except:
    import cv2


batchsize=3
dsl = cfsv
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

        print(temp_array.shape)

        # vs = np.vstack((temp_array,)+( imgs[j,0,:,:] for j in range(1,m)))

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


img = cv2.imread('/home/itrc/Desktop/Development/dense_flow_fbf/testfile-fbf/UCF101_images/ApplyLipstick/v_ApplyLipstick_g01_c02/image/image_0002.jpg')

img = [img]

imgs.shape

while True:

    imgs = ImgAugPipes(img)
    plt.imshow(imgs[0,0])



