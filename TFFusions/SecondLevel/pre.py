import tensorflow as tf
from TFFusions.toolkits.dataloader import getTrainItems,getValItems

# distill tf record
feature_file = '/mnt/md0/LSVC/feat_senet'

cnt = 0

second_level = []
with open('','r') as f:
    line = f.readline()
    second_level.append()

for item in getTrainItems():
    labels = item[1]

for item in getValItems():
    cnt+=1
