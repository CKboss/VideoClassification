from VideoClassification.utils.DataSetLoader.UCF101Loader import test_UCF101_ChooseOrderFromSameVideo, UCF101_TwoStream

from VideoClassification.utils.DataSetLoader.UCF101_DataSetLoader_FromFileName.PictureQueue import \
    GenVariables_VideoSpatialAndTemporal

dsl = test_UCF101_ChooseOrderFromSameVideo(dsl=UCF101_TwoStream)


def gen():
    return GenVariables_VideoSpatialAndTemporal(dsl=dsl, batchsize=4)


imgs, labels = gen()

print(id(imgs))
print(labels)
