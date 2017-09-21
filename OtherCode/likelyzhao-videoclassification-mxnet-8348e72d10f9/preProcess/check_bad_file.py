f = open('val_frame_single.lst')
import os
from multiprocessing import Pool

def checknow(path):
    import cv2
#    print(path)
    if  cv2.imread(path) is None:
        print("error " + path)

pool = Pool(processes=25)
for line in f.readlines():
    splits = line.strip().split('\t')
    filename = splits[len(splits)-1]
    filename = os.path.join('/workspace/data/data/',filename) 
    pool.apply_async(checknow,(filename,))
pool.close()
pool.join()
