
from VideoClassification.utils.DataSetLoader.UCF101Loader import ChooseRandomFromSameVideo,UCF101_TwoStream,UCF101_C3D,ChooseOrderFromSameVideo
import VideoClassification.Config.Config as Config

from importlib import reload

cfsv = ChooseOrderFromSameVideo(file=Config.Code_root+'/data/testlist01.txt', dsl=UCF101_TwoStream)

cfsv2 = ChooseRandomFromSameVideo(file=Config.Code_root+'/data/testlist01.txt', dsl=UCF101_TwoStream)

cfsv3 = ChooseRandomFromSameVideo(file=Config.Code_root+'/data/testlist01.txt', dsl=UCF101_C3D)


a,b = cfsv2[0]

filelists,lbs = cfsv2[1000]

lsts = list(zip(filelists,lbs))

if True:
    first_img = filelists[0]
    lsts = lsts[1:]

filelists = []

tims = [ int(files[0][0][-8:-4]) for files in lsts ]

ll = sorted(list(zip(tims,filelists)))

[l[1] for l in ll]

lbs


