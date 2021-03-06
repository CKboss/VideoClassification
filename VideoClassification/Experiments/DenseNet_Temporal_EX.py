import torch
import torch.nn as nn
from torch.autograd import Variable

try:
    from cv2 import cv2
except:
    import cv2

import VideoClassification.Config.Config as Config
from VideoClassification.model.densenet_twostream.densenet_twostream import dense169_temporalNet, dense201_temporalNet
from VideoClassification.utils.Others.Logger import Logger
from VideoClassification.utils.DataSetLoader.UCF101_DataSetLoader_FromFileName.UCF101Loader import \
    train_UCF0101_Temporal, test_UCF0101_Temporal
from VideoClassification.utils.Others.toolkits import accuracy, try_to_load_state_dict
from VideoClassification.utils.DataSetLoader.UCF101_DataSetLoader_FromFileName.PictureQueue import PictureQueue, \
    GenVariables_Temporal

############ Config

logger = Logger(Config.LOGSpace + Config.EX_ID)
savepath = Config.ExWorkSpace + Config.EX_ID + '/'

import os.path

if os.path.isdir(savepath) == False:
    os.mkdir(savepath)

batchsize = 64


############

def DenseNet161_temporal_Run():
    epochs = 81
    loops = 2001
    learningrate = 0.01
    attenuation = 0.1

    model = dense169_temporalNet(pretrained=False, dropout=0.4).cuda()

    if Config.LOAD_SAVED_MODE_PATH is not None:
        import types
        model.try_to_load_state_dict = types.MethodType(try_to_load_state_dict, model)
        model.try_to_load_state_dict(torch.load(Config.LOAD_SAVED_MODE_PATH))
        print('LOAD {} done!'.format(Config.LOAD_SAVED_MODE_PATH))

    lossfunc = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=learningrate)

    pq_train = PictureQueue(dsl=train_UCF0101_Temporal(), Gen=GenVariables_Temporal, batchsize=batchsize)
    pq_test = PictureQueue(dsl=test_UCF0101_Temporal(), Gen=GenVariables_Temporal, batchsize=batchsize)

    cnt = 0
    for epoch in range(epochs):

        for l in range(loops):

            cnt += 1

            imgs, labels = pq_train.Get()

            model.zero_grad()
            pred = model(imgs)
            loss = lossfunc(pred, labels)

            logger.scalar_summary('DenseNet161/Temporal/train_loss', loss.data[0], cnt)

            loss.backward()
            optim.step()

            print('Temporal epoch: {} cnt: {} loss: {}'.format(epoch, cnt, loss.data[0]))

            if cnt % 25 == 0:
                imgs, labels = pq_test.Get()
                pred = model.inference(imgs)
                loss = lossfunc(pred, labels)

                logger.scalar_summary('DenseNet161/Temporal/test_loss', loss.data[0], cnt)

                # acc
                acc = accuracy(pred, labels, topk=(1, 5, 10))
                logger.scalar_summary('DenseNet161/Temporal/test_acc@1', acc[0], cnt)
                logger.scalar_summary('DenseNet161/Temporal/test_acc@5', acc[1], cnt)
                logger.scalar_summary('DenseNet161/Temporal/test_acc@10', acc[2], cnt)

                imgs, labels = pq_train.Get()
                pred = model.inference(imgs)

                acc = accuracy(pred, labels, topk=(1, 5, 10))
                logger.scalar_summary('DenseNet161/Temporal/train_acc@1', acc[0], cnt)
                logger.scalar_summary('DenseNet161/Temporal/train_acc@5', acc[1], cnt)
                logger.scalar_summary('DenseNet161/Temporal/train_acc@10', acc[2], cnt)

            if cnt % 2000 == 0:
                savefile = savepath + 'DenseNet161_Temporal_EX1_{:02d}.pt'.format(epoch % 50)
                print('Temporal save model to {}'.format(savefile))
                torch.save(model.state_dict(), savefile)

        if epoch in [5, 10, 20, 50, 60]:
            learningrate = learningrate * attenuation
            optim = torch.optim.Adam(model.parameters(), lr=learningrate)


def DenseNet201_temporal_Run():
    epochs = 81
    loops = 2001
    learningrate = 0.0001
    attenuation = 0.1

    model = dense201_temporalNet(pretrained=False, dropout=0.9).cuda()

    if Config.LOAD_SAVED_MODE_PATH is not None:
        import types
        model.try_to_load_state_dict = types.MethodType(try_to_load_state_dict, model)
        model.try_to_load_state_dict(torch.load(Config.LOAD_SAVED_MODE_PATH))
        print('LOAD {} done!'.format(Config.LOAD_SAVED_MODE_PATH))

    lossfunc = nn.CrossEntropyLoss()
    optim = torch.optim.SGD(model.parameters(), lr=learningrate, momentum=0.1)

    pq_train = PictureQueue(dsl=train_UCF0101_Temporal(), Gen=GenVariables_Temporal, batchsize=batchsize)
    pq_test = PictureQueue(dsl=test_UCF0101_Temporal(), Gen=GenVariables_Temporal, batchsize=batchsize)

    cnt = 0
    for epoch in range(epochs):

        for l in range(loops):

            cnt += 1

            imgs, labels = pq_train.Get()

            model.zero_grad()
            pred = model(imgs)
            loss = lossfunc(pred, labels)

            logger.scalar_summary('DenseNet201/Temporal/train_loss', loss.data[0], cnt)

            loss.backward()
            optim.step()

            print('Temporal epoch: {} cnt: {} loss: {}'.format(epoch, cnt, loss.data[0]))
            if cnt % 25 == 0:
                imgs, labels = pq_test.Get()
                pred = model.inference(imgs)
                loss = lossfunc(pred, labels)

                logger.scalar_summary('DenseNet201/Temporal/test_loss', loss.data[0], cnt)

                # acc
                acc = accuracy(pred, labels, topk=(1, 5, 10))
                logger.scalar_summary('DenseNet201/Temporal/test_acc@1', acc[0], cnt)
                logger.scalar_summary('DenseNet201/Temporal/test_acc@5', acc[1], cnt)
                logger.scalar_summary('DenseNet201/Temporal/test_acc@10', acc[2], cnt)

                imgs, labels = pq_train.Get()
                pred = model.inference(imgs)

                acc = accuracy(pred, labels, topk=(1, 5, 10))
                logger.scalar_summary('DenseNet201/Temporal/train_acc@1', acc[0], cnt)
                logger.scalar_summary('DenseNet201/Temporal/train_acc@5', acc[1], cnt)
                logger.scalar_summary('DenseNet201/Temporal/train_acc@10', acc[2], cnt)

            if cnt % 2000 == 0:
                savefile = savepath + 'DenseNet201_Temporal_EX1_{:02d}.pt'.format(epoch % 50)
                print('Temporal save model to {}'.format(savefile))
                torch.save(model.state_dict(), savefile)

        if epoch in [10, 20, 50, 60]:
            learningrate = learningrate * attenuation
            optim = torch.optim.SGD(model.parameters(), lr=learningrate, momentum=0.9)


if __name__ == '__main__':
    x = torch.randn(3, 20, 224, 224)
    x = Variable(x)
