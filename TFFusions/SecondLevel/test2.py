import random
import numpy as np

from TFFusions.toolkits.dataloader import getTrainItems,getValItems,Load_Features_SENET,concurrent_get_items
from TFFusions.SecondLevel.GCForest import *

train_items = getTrainItems()

Load_Features_SENET(train_items[1][0],kind='train',limitlen=10)


rid = random.choices(list(range(len(train_items))),k=1000)

train_X = list()
train_Y = list()


for i in range(100):

    try:
        videolen,feature,label = concurrent_get_items(train_items[rid[i]],kind='train',load_func=Load_Features_SENET)
    except Exception as E:
        continue

    id = random.randint(0,videolen)
    randlen = random.choices(list(range(videolen)),k=1)

    f = list()
    for j in randlen:
        f.append(feature[j,:])
    f = np.array(f)
    f = np.reshape(f,(-1))

    train_X.append(f)
    train_Y.append(label[0])

train_X = np.array(train_X)
train_Y = np.array(train_Y)



gcf = gcForest(shape_1X=512, window=4, tolerance=0.2, n_jobs=10)
gcf.fit(train_X,train_Y)

