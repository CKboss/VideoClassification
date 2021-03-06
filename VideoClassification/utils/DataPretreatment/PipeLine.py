from numba import jit
import numpy as np
from functools import lru_cache
import random
import bisect

try:
    from cv2 import cv2
except:
    import cv2

from imgaug import augmenters as iaa
import imgaug as ia

"""图像预处理管线
各个部分通过img连接
"""


@lru_cache(maxsize=8192)
def Imread_lru(imgfilepath):
    return cv2.imread(imgfilepath)


@jit
def fitToPytorch(img):
    '''
    fit img to pytorch : channel ( 3 ) x height x width
    :param img:
    :return:
    '''
    if len(img.shape) == 2:
        return img
    elif len(img.shape) == 3:
        ret = np.transpose(img, [2, 0, 1])
        return ret
    else:
        raise NotImplementedError


@jit
def ReSize(img, outshape=(228, 228)):
    return cv2.resize(img, outshape)


@jit
def ToBlackAndWhite(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


@jit
def Identify(img, outshape=(228, 228)):
    '''
    :param img: input image
    :return: output image
    '''
    # 二值化
    img = ToBlackAndWhite(img)
    img = ReSize(img, outshape)
    return img


@jit
def CentreCut(img, outshape=(228, 228)):
    (h, w) = outshape
    (n, m) = img.shape

    assert n > h, 'n must great than h'
    assert m > w, 'm must great than w'

    m_x = int(n / 2)
    m_y = int(m / 2)

    dx = h / 2
    dy = w / 2

    l = [m_x - dx, m_x + dx - 1, m_y - dy, m_y + dy - 1]

    l = list(map(lambda x: int(x), l))

    return ReSize(img[l[0]:l[1], l[2]:l[3]], outshape)


@jit
def RandFlipLR(img):
    return iaa.Fliplr(0.5).augment_image(img)


@jit
def RandFlipUD(img):
    return iaa.Flipud(0.5).augment_image(img)


@jit
def RandAdd(img):
    return iaa.Add((-10, 10)).augment_image(img)


@jit
def RandMultiply(img):
    return iaa.Multiply((0.9, 1.1)).augment_image(img)


@jit
def RandGrayscale(img):
    return iaa.Grayscale((0, 1)).augment_image(img)


@jit
def RandCut(img):
    return iaa.Crop(percent=(0, 0.15)).augment_image(img)


@jit
def RandElastic(img):
    return iaa.ElasticTransformation(alpha=(0, 0.1), sigma=(0, 0.2)).augment_image(img)


@jit
def RandGaussBlur(img):
    return iaa.Grayscale(alpha=(0, 0.5)).augment_image(img)


def PipeLineRun(img, funcs, params, **kwargs):
    # assert len(funcs) == len(params), 'func and params showld have same length'
    Img = img.copy()
    for i, func in enumerate(funcs):
        try:
            if params[i] is None:
                Img = func(Img, **kwargs)
            else:
                Img = func(Img, **params[i], **kwargs)
        except Exception as e:
            print(e)
            print('in PipeLineRun func: ', func, ' params: ', **params[i])

    return Img


def DefaultPipe(img):
    '''
    默认PipeLine什么都不做
    :param img:
    :return: img
    '''
    return img


@jit
def FlipLR(img, flag):
    if flag == True:
        return cv2.flip(img, 1)
    return img


@jit
def FlipUD(img, flag):
    if flag == True:
        return cv2.flip(img, 0)
    return img


@jit
def CutImg(img, kind, kindw):
    w = [250, 220, 230, 240]
    w = w[kindw]

    cr = [[0, w, 0, w],
          [255 - w, 255, 0, w],
          [0, w, 339 - w, 339],
          [255 - w, 255, 339 - w, 339],
          [255 - w, 255, 339 - w, 339],
          # [w//2,225-w//2,w//2,339-w//2]
          ]

    cr = cr[kind]

    # if kind == 4:
    #     if cr[0] >= cr[1]:
    #         cr[0],cr[1] = cr[1],cr[0]
    #     if cr[2] >= cr[3]:
    #         cr[2],cr[3] = cr[2],cr[3]

    img = img[cr[0]:cr[1], cr[2]:cr[3]]
    return img


@jit
def imadjust(src, tol=0, vin=[0, 255], vout=(0, 255)):
    # 删除了tol参数
    # src : input one-layer image (numpy array)
    # tol : tolerance, from 0 to 100.
    # vin  : src image bounds
    # vout : dst image bounds
    # return : output img

    assert len(src.shape) == 2, 'Input image should be 2-dims'

    tol = max(0, min(100, tol))

    if tol > 0:
        raise NotImplementedError
    # # Compute in and out limits
    #     # Histogram
    #     hist = np.histogram(src,bins=list(range(256)),range=(0,255))[0]
    #
    #     # Cumulative histogram
    #     cum = np.cumsum(hist)
    #
    #     # Compute bounds
    #     total = src.shape[0] * src.shape[1]
    #     low_bound = total * tol / 100
    #     upp_bound = total * (100 - tol) / 100
    #     vin[0] = bisect.bisect_left(cum, low_bound)
    #     vin[1] = bisect.bisect_left(cum, upp_bound)

    # Stretching
    scale = (vout[1] - vout[0]) / (vin[1] - vin[0])
    vs = src - vin[0]
    vs[src < vin[0]] = 0
    vd = vs * scale + 0.5 + vout[0]
    vd[vd > vout[1]] = vout[1]
    dst = vd

    return dst


@jit
def hisEqul(img):
    return cv2.equalizeHist(img)


@jit
def hisEqulColor(img):
    if len(img.shape) == 2:
        return hisEqul(img)

    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    channels = cv2.split(ycrcb)
    cv2.equalizeHist(channels[0], channels[0])
    cv2.merge(channels, ycrcb)
    cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR, img)
    return img


@jit
def Normalize(img, Norm=True, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
    img = img / 255.
    # img = img.astype(np.float32)
    # if np.max(img) - np.min(img) > 0.001:
    #     img = (img-np.min(img)) / (np.max(img)-np.min(img))
    # else:
    #     img = img - np.min(img)
    if Norm:
        level = len(img.shape)
        assert level <= 3, 'level should <= 3'
        if level == 3:
            for i in range(level):
                img[:, :, i] = (img[:, :, i] - mean[i]) / std[i]
        else:
            img[:, :] = (img[:, :] - mean[0]) / std[0]

    return img


# def ImgAugPipes(imgs,isTemporal=False,outputshape=(224,224),isNormal=True,**kwargs):

# Gen Paramer
# p1 = random.choice([True,False])
# p2 = random.choice([True,False])
# p3 = random.choice([0,1,2,3,4])
# p4 = random.choice([0,1,2,3])

# if isNormal==True:
#     ParamerList = [(ReSize,{'outshape':(256,256)}),
#                    (FlipLR,{'flag':p1}),
#                    # (FlipUD,{'flag':p2}),
#                    (CutImg,{'kind':p3,'kindw':p4}),
#                    (ReSize,{'outshape':outputshape}),
#                    (Normalize,{'Norm':True}),
#                    (fitToPytorch,None)]
# else:
#     ParamerList = [(ReSize,{'outshape':(256,256)}),
#                    (FlipLR,{'flag':p1}),
#                    # (FlipUD,{'flag':p2}),
#                    (CutImg,{'kind':p3,'kindw':p4}),
#                    (ReSize,{'outshape':outputshape}),
#                    (fitToPytorch,None)]
#
#
# if isTemporal==True:
#     ParamerList = [(ToBlackAndWhite,None),
#                    # (hisEqulColor,None),
#                    ] + ParamerList

# Run in PipeLine

# print(funcs)
# print(params)
# rets = []
# for img in imgs:
#     Img = PipeLineRun(img,funcs,params)
#     rets.append(Img)
# return np.array(rets)


def ImgAugPipes(imgs, isTemporal=False, outputshape=(224, 224), isNormal=True, NoAug=False, **kwargs):
    # 数据增强
    ParamerList = [
        # (RandFlipUD,None),
        (RandFlipLR, None),
        # (RandAdd,None),
        (RandMultiply, None),
        (RandCut, None),
        (RandElastic, None),
    ]
    funcs = [x[0] for x in ParamerList]
    params = [x[1] for x in ParamerList]
    img = imgs[0].copy()

    n = len(imgs)
    img = np.concatenate([ReSize(imgs[i], outshape=(256, 256)) for i in range(n)], axis=2)

    #
    # for i in range(1,n):
    #     try:
    #         img = np.concatenate((img,imgs[i]),axis=2)
    #     except Exception as E:
    #         print(E)
    #         print('img_shape:',img.shape)
    #         for j in range(n):
    #             print(j,'->',imgs[j].shape)
    #         raise RuntimeError

    if NoAug == False:
        img = PipeLineRun(img, funcs, params)
    img = ReSize(img, outputshape)

    # split img into imgs
    imgs = []
    for i in range(n):
        part = img[:, :, i * 3:i * 3 + 3]
        imgs.append(part.copy())

    if isNormal == True:
        for i in range(n):
            imgs[i] = Normalize(imgs[i])

    if isTemporal == True:
        for i in range(n):
            imgs[i] = imgs[i][:, :, 0]
        imgs = np.array(imgs)
        return imgs
    else:
        # fit to torch
        ret = np.array(imgs)
        # ret = fitToPytorch(ret)
        ret = np.transpose(ret, (0, 3, 2, 1))

    return ret


def GenTensors(imgpathss, **kwargs):
    import torch
    imgss = []
    for imgpaths in imgpathss:

        imgs = []

        for path in imgpaths:
            imgs.append(cv2.imread(path))

        imgs = np.array(imgs)
        imgs = ImgAugPipes(imgs, **kwargs)

        imgss.append(imgs)

    imgss = np.array(imgss)
    return torch.from_numpy(imgss).float()


if __name__ == '__main__':

    imgpaths = [
        '/home/lab/Desktop/Development/dense_flow_fbf/testfile-fbf/UCF101_images/ApplyLipstick/v_ApplyLipstick_g01_c02/flow_x/flow_x_0063.jpg',
        '/home/lab/Desktop/Development/dense_flow_fbf/testfile-fbf/UCF101_images/ApplyLipstick/v_ApplyLipstick_g01_c02/flow_x/flow_x_0064.jpg', ]

    imgpaths = imgpaths + imgpaths

    imgs = []
    for path in imgpaths:
        imgs.append(cv2.imread(path))
    imgs = np.array(imgs)

    id(imgs)
    type(imgs)

    for i in range(1000):
        aimgs = ImgAugPipes(imgs)
        print(aimgs.shape)

    imgpathss = [imgpaths, imgpaths, imgpaths]
    timgs = GenTensors(imgpathss, isTemporal=True)
