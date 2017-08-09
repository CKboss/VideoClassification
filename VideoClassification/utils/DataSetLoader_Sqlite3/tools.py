import os
import os.path
import sys

import VideoClassification.Config.Config as Config
from VideoClassification.utils.DataSetLoader.UCF101Loader import GaoImageID,image_id

# insert image to dataset


for root,dirs,files in os.walk(Config.UCF101_images_root):
    if len(files) > 0:
        for file in files:
            print(root)
            print(dir)

line = '/datacenter/1/UCF_Data/UCF101_images/FrisbeeCatch/v_FrisbeeCatch_g01_c01/flow_x'

def chuli(line):
    GaoImageID()
    items = line.split('/')
    classname = items[-3]
    label = image_id[classname]
    videoname = items[-2]
    if items[-1][:4] == 'flow':
        kind = 'flow'
    else :
        kind = 'frame'
    pass

