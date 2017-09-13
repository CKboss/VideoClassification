import numpy as np
import xgboost as xgb

from TFFusions.toolkits.dataloader import getTrainItems,getValItems,Load_Features_SENET,concurrent_get_items

trainval_data = '/mnt/md0/LSVC/feature_1fps_vid/trainval/'
test_data = '/mnt/md0/LSVC/feature_1fps_vid/test/'
filename_suffix = '{}_fc6_vgg19_video.binary'

train_items = getTrainItems()


loop = 100
train_X = []
# train_Y = np.zeros((loop,500))
train_Y = []

for i in range(loop):
    filename = trainval_data+filename_suffix.format(train_items[i][0])
    train_X.append(np.fromfile(filename,dtype=np.float32))
    # train_Y[i,train_items[i][1][0]] = 1
    train_Y.append(train_items[i][1][0])

train_X = np.array(train_X)
# train_Y = train_Y.astype(np.int32)
train_Y = np.array(train_Y)

xg_train = xgb.DMatrix(train_X,label=train_Y)

xgb_params = {
    'eta': 0.05,
    'max_depth': 5,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'multi:softmax',
    'eval_metric': 'rmse',
    'silent': 1,
    'num_class': 500,
}


tree = xgb.train(params=xgb_params,dtrain=xg_train,num_boost_round=100,early_stopping_rounds=10)

tree.predict(xg_train)

