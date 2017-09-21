import cv2
import os
from util.listfile import listfile

PATHUCF = '/workspace/data/UCF-101'
PATHOUT = '/workspace/data/frame/'


def _mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


avifilelist, dictName = listfile(PATHUCF, "avi", True)
_mkdir(PATHOUT)
for avifilename in avifilelist:
    cap = cv2.VideoCapture(avifilename)

    count = 0
    while (1):
        ret, frame = cap.read()
        if ret == False:
            break
        #        print(frame.shape)
        basename = os.path.basename(avifilename)
        dirname = os.path.dirname(avifilename)
        splits = dirname.split('/')
        labelpath = splits[len(splits) - 1]
        filename = os.path.splitext(basename)[0]
        #        print labelpath
        savepath = os.path.join(PATHOUT, labelpath)
        _mkdir(savepath)
        #        print(str(count))
        savename = savepath + '/' + filename + "_" + str(count) + ".jpg"
        #        print(filename)
        #        print(savename)
        cv2.imwrite(savename, frame)
        count += 1
    print(count)
    cap.release()
