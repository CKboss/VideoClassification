try:
    from cv2 import cv2
except:
    import cv2
import numpy as np

import sqlite3
from sklearn.svm import LinearSVC
import types

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.models.vgg import vgg16

import VideoClassification.Config.Config as Config
from VideoClassification.utils.DataSetLoader.PictureQueue import ImgAugPipes

# load midfeature

savepath = Config.vgg16pretrainfile

model = vgg16().cuda()
model.load_state_dict(torch.load(savepath))

new_classifier = list(model.classifier.children())
new_classifier = new_classifier[:4]
new_classifier = nn.Sequential(*new_classifier)
model.classifier = new_classifier

# use SVM

conn = sqlite3.connect(Config.Code_root+'data/UCF101.db')
cusor = conn.cursor()

cls = LinearSVC()
cusor.execute('SELECT imgpath,label FROM ImgSets WHERE imgkind="frame" and splitkind="train" ORDER BY RANDOM() LIMIT 10000;')

for i in range(200):

    items = cusor.fetchmany(50)

    imgpaths = [ Config.UCF101_images_root+item[0] for item in items]
    labels = [ item[1] for item in items]
    imgs = list(map(cv2.imread, imgpaths))
    imgs = np.array(ImgAugPipes(imgs))

    imgs = Variable(torch.from_numpy(imgs)).float().cuda()
    labels = Variable(torch.from_numpy(np.array(labels))).long().cuda()

    pred = model(imgs)
    pred = pred.cpu().data.numpy()



# get ACC



if __name__=='__main__':
    x = torch.randn(4,3,224,224)
    x = Variable(x).cuda()
    y = model(x)
    y.size()
