import numpy as np
from numba import jit
import os
from TFFusions.toolkits.dataloader import getTrainItems, getTestItems, getValItems
import TFFusions.Config.Config as Config

# read all value and calu mean

def lf(videoname=None, kind=None, limitlen=600):
    if kind == 'train':
        prefix = Config.DATA_PATH + 'trainval/'
    elif kind == 'val':
        prefix = Config.DATA_PATH + 'trainval/'
    elif kind == 'test':
        prefix = Config.DATA_PATH + 'test/'
    else:
        raise NotImplementedError
    # videoname example : lsvc000000
    filename = prefix + '{}_fc6_vgg19_frame.binary'.format(videoname)
    frame_features = np.fromfile(filename, dtype='float32')
    return frame_features

valitems = getValItems()
trainitems = getTrainItems()
testitems = getTestItems()


@jit
def online_variance(data):
    n = 0
    mean = 0.0
    M2 = 0.0
    for x in data:
        n += 1
        delta = x - mean
        mean += delta/n
        delta2 = x - mean
        M2 += delta*delta2
    return M2 / n , mean


class OnlineVarMean(object):
    def __init__(self):
        self.n = 0
        self.mean = 0
        self.M2 = 0
    @jit
    def Update(self,x):
        self.n += 1
        delta = x - self.mean
        self.mean += delta/self.n
        delta2 = x - self.mean
        self.M2 += delta*delta2
    @jit
    def GetVarMean(self):
        return self.M2/self.n, self.mean

ovm = OnlineVarMean()

cnt=0

print('begin')

for item in valitems:
    feature = lf(item[0],'val')
    cnt+=1
    for x in feature.tolist():
        ovm.Update(x)
    if cnt%100 == 0:
        print(cnt)
print(ovm.GetVarMean())

for item in testitems:
    feature = lf(item[0],'test')
    cnt+=1
    for x in feature.tolist():
        ovm.Update(x)
    if cnt%100 == 0:
        print(cnt)
print(ovm.GetVarMean())

for item in trainitems:
    feature = lf(item[0],'train')
    cnt+=1
    for x in feature.tolist():
        ovm.Update(x)
    if cnt%100 == 0:
        print(cnt)
print(ovm.GetVarMean())

with open('/tmp/test_val.txt','w') as f:
    f.write(str(ovm.GetVarMean()))



# calu Var

