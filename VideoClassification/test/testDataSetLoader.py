
from VideoClassification.utils.DataSetLoader.UCF101Loader import ChooseFromSameVideo,UCF101_TwoStream
import VideoClassification.Config.Config as Config


cfsv = ChooseFromSameVideo(file=Config.Code_root+'/data/testlist01.txt',
                           dsl=UCF101_TwoStream)


a,b = cfsv[0]
