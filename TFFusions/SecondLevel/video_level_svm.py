import numpy as np
import pickle

from TFFusions.toolkits.dataloader import getTrainItems,getValItems,Load_Features_SENET,concurrent_get_items

trainval_data = '/mnt/md0/LSVC/feature_1fps_vid/trainval/'
test_data = '/mnt/md0/LSVC/feature_1fps_vid/test/'
filename_suffix = '{}_fc6_vgg19_video.binary'

# train_items = getTrainItems()
# loop = len(train_items)
# train_X = []
# train_Y = []
# for i in range(loop):
#     try:
#         filename = trainval_data+filename_suffix.format(train_items[i][0])
#         train_X.append(np.fromfile(filename,dtype=np.float32))
#         train_Y.append(train_items[i][1][0])
#         if i%100 == 0:
#             print(i)
#     except Exception as E:
#         pass
# train_X = np.array(train_X)
# train_Y = np.array(train_Y)
#
# np.savez('/mnt/md0/LSVC/downloads/train_XY',train_X=train_X,train_Y=train_Y)
#
# val_items = getValItems()
# val_X = []
# val_Y = []
# loop = len(val_items)
# for i in range(loop):
#     try:
#         filename = trainval_data+filename_suffix.format(train_items[i][0])
#         val_X.append(np.fromfile(filename,dtype=np.float32))
#         val_Y.append(train_items[i][1][0])
#         if i%100 == 0:
#             print(i)
#     except Exception as E:
#         pass
# val_X = np.array(val_X)
# val_Y = np.array(val_Y)
#
# np.savez('/mnt/md0/LSVC/downloads/val_XY',train_X=val_X,train_Y=val_Y)

data1 = np.load('/mnt/md0/LSVC/downloads/train_XY.npz')
train_X,train_Y = data1['train_X'],data1['train_Y']
data1 = np.load('/mnt/md0/LSVC/downloads/val_XY.npz')
val_X,val_Y = data1['train_X'],data1['train_Y']

from sklearn.svm import SVC

clf = SVC(probability=True)

clf.fit(train_X,train_Y)

pickle.dump(clf,open('/tmp/t1.pkl','wb'),True)

# pred = clf.predict(val_X)

# print(pred.shape)

print('over')
