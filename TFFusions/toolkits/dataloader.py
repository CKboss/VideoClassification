import numpy as np

import TFFusions.Config.Config as Config

classId = dict()

train_items = None
test_items = None
val_items = None

def _load_index(filename=Config.INDEX_DATA):
    with open(filename,'r') as f:
        for line in f.readlines():
            line = line.replace(' ','')
            line = line.replace('\n','')
            line = line.split('\t')
            classId[line[1]] = int(line[0])-1

def _load_data(filename):
    # 最多4个标签
    ret = []
    with open(filename,'r') as f:
        for line in f.readlines():
            line = line.replace('\n','')
            items = line.split(',')
            name = items[0]
            value = list(map(lambda x: int(x)-1 , items[1:]))
            ret.append((name,value))
    return ret

def _load_labels():

    global train_items
    global test_items
    global val_items

    _load_index(Config.INDEX_DATA)
    train_items = _load_data(Config.TRAIN_DATA)
    test_items = _load_data(Config.TEST_DATA)
    val_items = _load_data(Config.VAL_DATA)


def getTrainItems():
    global train_items
    if train_items is None:
        _load_labels()
    return train_items

def getValItems():
    global val_items
    if val_items is None:
        _load_labels()
    return val_items

def getTestItems():
    global test_items
    if test_items is None:
        _load_labels()
    return train_items

def getClassId():
    global classId
    if test_items == 0 :
        _load_labels()
    return classId

def Load_Features(videoname=None,kind=None,limitlen=600):
    if kind == 'train':
        prefix = Config.DATA_PATH+'trainval/'
    elif kind == 'val':
        prefix = Config.DATA_PATH+'trainval/'
    elif kind == 'test':
        prefix = Config.DATA_PATH+'test/'
    else:
        raise NotImplementedError
    # videoname example : lsvc000000
    filename = prefix+'{}_fc6_vgg19_frame.binary'.format(videoname)
    frame_features = np.fromfile(filename, dtype='float32').reshape(-1, 4096)

    # limit the frames len to limitlen
    if frame_features.shape[0] > limitlen:
        h = (frame_features.shape[0] - limitlen) // 2
        frame_features = frame_features[h+1:-h-1,:]
    return frame_features

# 长度3~700之间

if __name__=='__main__':
    _load_labels()
    # feature = Load_Features(videoname='lsvc000008')
    import random
    ll = []
    for i in random.choices(getTrainItems(),k=3000):
        videoname = i[0]
        feature = Load_Features(videoname=videoname,kind='train')
        # print(feature.shape[0])
        ll.append(feature.shape[0])
