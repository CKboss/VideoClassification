import numpy as np
import concurrent.futures
import multiprocessing
from multiprocessing import Process
import random
import time

import TFFusions.Config.Config as Config

classId = dict()

train_items = None
test_items = None
val_items = None


def _load_index(filename=Config.INDEX_DATA):
    with open(filename, 'r') as f:
        for line in f.readlines():
            line = line.replace(' ', '')
            line = line.replace('\n', '')
            line = line.split('\t')
            classId[line[1]] = int(line[0]) - 1


def _load_data(filename):
    # 最多4个标签
    ret = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            line = line.replace('\n', '')
            items = line.split(',')
            name = items[0]
            value = list(map(lambda x: int(x) - 1, items[1:]))
            ret.append((name, value))
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
    return test_items


def getClassId():
    global classId
    if test_items == 0:
        _load_labels()
    return classId

def Load_Features_INC(videoname=None,kind=None,limitlen=600):
    if kind == 'train' or kind=='val':
        prefix = Config.DATA_PATH + 'inc_trainval/'
    else:
        raise NotImplementedError
    # videoname example : lsvc000000
    filename = prefix + '{}_flatten_imagenet22k_frame.binary'.format(videoname)
    frame_features = np.fromfile(filename,dtype='float32').reshape(-1,1024)
    # limit the frames len to limitlen
    if frame_features.shape[0] > limitlen:
        h = (frame_features.shape[0] - limitlen) // 2
        frame_features = frame_features[h + 1:-h - 1, :]
    return frame_features

def Load_Features_SENET(videoname=None, kind=None, limitlen=600):
    prefix = Config.DATA_PATH + 'feat_senet/'
    # videoname example : lsvc000000
    filename = prefix + '{}_pool5_senet.binary'.format(videoname)
    frame_features = np.fromfile(filename, dtype='float32').reshape(-1, 2048)
    # limit the frames len to limitlen
    if frame_features.shape[0] > limitlen:
        h = (frame_features.shape[0] - limitlen) // 2
        frame_features = frame_features[h + 1:-h - 1, :]
    return frame_features

def Load_Features(videoname=None, kind=None, limitlen=600):
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
    frame_features = np.fromfile(filename, dtype='float32').reshape(-1, 4096)

    # limit the frames len to limitlen
    if frame_features.shape[0] > limitlen:
        h = (frame_features.shape[0] - limitlen) // 2
        frame_features = frame_features[h + 1:-h - 1, :]
    return frame_features


def concurrent_get_items(item, kind , load_func=Load_Features):
    """
    :param item: a tuple ('video_name',[labels])
    :param kind: from which kinds of datasets [train/test/val]
    :return: a tuple the network need
    """
    # try:
    #     feat = Load_Features(item[0],kind=kind,limitlen=600)
    # except Exception as E:
    #     print(E)
    #     item = ('lsvc000191', [222])
    #     feat = Load_Features(item[0],kind=kind,limitlen=600)

    feat = load_func(item[0], kind=kind, limitlen=600)
    ax0_len = feat.shape[0]
    return (ax0_len, feat, item[1])


GLOBAL_EXECUTOR = None


def GetExecutor():
    global GLOBAL_EXECUTOR
    if GLOBAL_EXECUTOR is None:
        GLOBAL_EXECUTOR = concurrent.futures.ProcessPoolExecutor()
    return GLOBAL_EXECUTOR


def gen_tf_input(items, kind):
    features = []
    video_frames = []
    labels = []

    kinds = [kind for __ in range(len(items))]
    executor = GetExecutor()
    for RET in executor.map(concurrent_get_items, items, kinds):
        video_frames.append(RET[0])
        features.append(RET[1])
        labels.append(RET[2])

    batchsize = len(labels)
    features_zeros = np.zeros((batchsize, 600, 4096))
    for i in range(batchsize):
        features_zeros[i, :features[i].shape[0], :] = features[i]

    target_label = np.zeros((len(items), 500))
    for id, label in enumerate(labels):
        for la in label:
            target_label[id, la] = 1

    return features_zeros, video_frames, target_label


class PictureQueue(object):
    def __init__(self, kind, batchsize=8, worker=5, mxsize=10):

        self.kind = kind
        if kind == 'train':
            self.datasets = getTrainItems()
        elif kind == 'test':
            self.datasets = getTestItems()
        elif kind == 'val':
            self.datasets = getValItems()
        self.worker = worker

        # use manager to share queue between process
        self.manager = multiprocessing.Manager()
        self.q = self.manager.Queue(mxsize)

        self.batchsize = batchsize

        self.mainQ = self.manager.Queue(maxsize=mxsize * 2)
        self.q = [self.manager.Queue(maxsize=mxsize) for i in range(worker)]
        self.ps = []

        for i in range(worker):
            self.ps.append(Process(target=self.pr, args=(self.q[i],), name='Producter_{}'.format(i)))

        # self.ps.append(Process(target=self.collectorPr,name='CollectorPR'))

        for i in range(worker):
            self.ps[i].deamon = True
            self.ps[i].start()

    def collectorPr(self):
        time.sleep(1)
        for i in range(self.worker):
            # print('collect pr:',self.mainQ.qsize(),'try to get from queue: ',i,' which size is ',self.q[i].qsize())
            if self.mainQ.full():
                break
            while True:
                try:
                    item = self.q[i].get_nowait()
                    # print(' got item !')
                    self.mainQ.put_nowait(item)
                    # print(' push in it ')
                except Exception as E:
                    break

    def pr(self, que):
        while True:
            if que.full():
                time.sleep(1)
                continue
            items = random.choices(self.datasets, k=self.batchsize)
            features = []
            video_frames = []
            labels = []
            for item in items:
                RET = concurrent_get_items(item, self.kind)
                video_frames.append(RET[0])
                features.append(RET[1])
                labels.append(RET[2])
            batchsize = self.batchsize
            features_zeros = np.zeros((batchsize, 600, 4096))
            for i in range(batchsize):
                features_zeros[i, :features[i].shape[0], :] = features[i]
            target_label = np.zeros((len(items), 500))
            for id, label in enumerate(labels):
                for la in label:
                    target_label[id, la] = 1

            processname = multiprocessing.current_process().name
            # print('{}: {}'.format(processname,que.qsize()))
            que.put((features_zeros, video_frames, target_label))

    def Get(self):
        # print(Process.pid,'Try To Get q.size:',self.q.qsize())
        while True:
            try:
                return self.mainQ.get_nowait()
            except:
                self.collectorPr()


# 长度3~700之间

if __name__ == '__main__':
    import time

    train_items = getTrainItems()

    items = train_items[:3]
    Load_Features_INC(items[2][0],'train')

    items = train_items[:64]

    queue = PictureQueue(kind='train', batchsize=16, worker=5, mxsize=3)
    item = queue.Get()


    # item = items[0]
    #
    # a,b,c = concurrent_get_items(item,'train')
    #
    # a = time.time()
    # x,y,z = gen_tf_input(items,'train')
    # b = time.time()
    # print(b-a)

    # _load_labels()
    # # feature = Load_Features(videoname='lsvc000008')
    # import random
    # ll = []
    # for i in random.choices(getTrainItems(),k=3000):
    #     videoname = i[0]
    #     feature = Load_Features(videoname=videoname,kind='train')
    #     # print(feature.shape[0])
    #     ll.append(feature.shape[0])
