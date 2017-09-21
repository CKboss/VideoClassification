import os
import shutil
from  util.listfile import listfile
def _mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

PATH = '/workspace/data/frame/'
PATHTEST = '/workspace/data/test/'
f = open('/workspace/data/ucfTrainTestlist/trainlist01.txt')
for line in f.readlines():
    avifilename = line.strip()
    basename = os.path.basename(avifilename)
    dirname = os.path.dirname(avifilename)
    splits = dirname.split('/')
    labelpath = splits[len(splits)-1]
    filename = os.path.splitext(basename)[0] 
    jpgpath = os.path.join(PATH,labelpath)
    filelist,_ = listfile(jpgpath,'jpg')
    _mkdir(os.path.join(PATHTEST,labelpath))
    for file in filelist:
        if filename in  file:
            dstpath = file.replace(PATH,PATHTEST)
            shutil.move(file,dstpath)
    #print(jpgpath)
    print(filename)
f.close()
