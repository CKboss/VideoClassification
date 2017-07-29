from numba import jit
import numpy as np
from functools import lru_cache

try:
    from cv2 import cv2
except:
    import cv2

from imgaug import augmenters as iaa
import imgaug as ia

"""图像预处理管线
各个部分通过img连接
"""

@lru_cache(maxsize=4096)
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
        ret = np.zeros(((3,)+img.shape))
        for i in range(3):
            ret[i] = img
        return ret
    elif len(img.shape) == 3:
        ret = np.transpose(img,[2,0,1])
        return ret
    else:
        raise NotImplementedError

@jit
def ReSize(img,outshape=(228,228)):
    return cv2.resize(img,outshape)

@jit
def ToBlackAndWhite(img):
    return cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

@jit
def Identify(img,outshape=(228,228)):
    '''
    :param img: input image
    :return: output image
    '''
    # 二值化
    img = ToBlackAndWhite(img)
    img = ReSize(img,outshape)
    return img

@jit
def CentreCut(img,outshape=(228,228)):

    (h,w) = outshape
    (n,m) = img.shape

    assert n>h , 'n must great than h'
    assert m>w , 'm must great than w'

    m_x = int(n/2)
    m_y = int(m/2)

    dx = h/2
    dy = w/2

    l = [m_x-dx,m_x+dx-1,m_y-dy,m_y+dy-1]

    l = list(map(lambda x:int(x),l))

    return ReSize(img[l[0]:l[1],l[2]:l[3]],outshape)


@jit
def RandFlipLR(img):
    return iaa.Fliplr(0.5).augment_image(img)

@jit
def RandFlipUD(img):
    return iaa.Flipud(0.5).augment_image(img)

@jit
def RandAdd(img):
    return iaa.Add((-10,10)).augment_image(img)

@jit
def RandMultiply(img):
    return iaa.Multiply((0.9,1.1)).augment_image(img)

@jit
def RandGrayscale(img):
    return iaa.Grayscale((0,1)).augment_image(img)

@jit
def RandCut(img):
    return iaa.Crop(percent=(0,0.15)).augment_image(img)

@jit
def RandElastic(img):
    return iaa.ElasticTransformation(alpha=(0,0.1),sigma=(0,0.2)).augment_image(img)

@jit
def RandGaussBlur(img):
    return iaa.Grayscale(alpha=(0,0.5)).augment_image(img)

def PipeLineRun(img,funcs,params):

    #assert len(funcs) == len(params), 'func and params showld have same length'
    Img = img.copy()
    for i,func in enumerate(funcs):
        if params[i] is None:
            Img = func(Img)
        else :
            Img = func(Img,**params[i])
    return Img

def DefaultPipe(img):
    '''
    默认PipeLine什么都不做
    :param img:
    :return: img
    '''
    return img



