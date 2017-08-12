import os
import os.path

import VideoClassification.Config.Config as Config
from VideoClassification.utils.DBTools.MySqlTools.dbSQL import INSERT_NEW_IMAGE, INSERT_VIDEO_LABELS
from VideoClassification.utils.DBTools.MySqlTools.dbcore import ConnPool
from VideoClassification.utils.DataSetLoader.UCF101_DataSetLoader_FromFileName.UCF101Loader import GaoImageID, image_id

testlst = []

def getTrain_dict():
    if len(testlst) == 0:
        with open(Config.Code_root+'/data/testlist01.txt','r') as f:
            for line in f.readlines():
                videoname = line.split(' ')[0]
                videoname = videoname[:-5]
                testlst.append(videoname.split('/')[1])


line = '/datacenter/1/UCF_Data/UCF101_images/FrisbeeCatch/v_FrisbeeCatch_g01_c01/flow_x'

def chuli(line):

    items = line.split('/')
    classname = items[-3]
    label = image_id[classname]
    videoname = items[-2]

    if videoname in testlst:
        splitkind = 'test'
    else:
        splitkind = 'train'

    if items[-1][:4] == 'flow':
        imgkind = 'flow'
    else :
        imgkind = 'frame'

    pathprefix = '/'.join(items[5:])

    return splitkind,imgkind,videoname,label,pathprefix


# chuli(line)


# , splitkind [train/test/val] , imgfilepath [] , imgfilename [] \
#     imgkind [frame/optial] , video_name , label , ord

def FindAndInsertImages():

    GaoImageID()
    getTrain_dict()

    conn = ConnPool.connect()
    cursor = conn.cursor()

    for root,dirs,files in os.walk(Config.UCF101_images_root):
        if len(files) > 0:
            print('--> ',root)
            need_to_insert = []
            for file in files:
                splitkind,imgkind,videoname,label,prefixpath = chuli(root)
                imgname = file
                ord = int(imgname[:-4].split('_')[-1])*2
                if imgkind=='flow' and imgname[:6]=='flow_y':
                    ord = ord+1
                imgpath = prefixpath + '/' + file
                need_to_insert.append([splitkind,imgpath,imgname,imgkind,videoname,label,ord])
            try :
                cursor.executemany(INSERT_NEW_IMAGE,need_to_insert)
                cursor.execute(INSERT_VIDEO_LABELS,(videoname,label))
            except Exception as E:
                print('ERROR!',E)


def getFrames_imgfilepath(splitkind='test'):

    '''
    :return: 连续的原始图片
    '''
    sql1 = 'SELECT imgname,videoname,ord FROM ImgSets WHERE splitkind="{}" and imgkind="frame" ORDER BY RAND(),videoname,RAND(),ord LIMIT 1;'.format(splitkind)

    conn = ConnPool.connect()
    cursor = conn.cursor()
    cursor.execute(sql1)
    res = cursor.fetchone()

    # print(res)
    videoname = res[1]
    ord = res[2]

    sql2 = 'SELECT imgpath,first_label FROM ImgSets WHERE videoname="{}" and splitkind="{}" and imgkind="frame" and ord>={} and ord<{} ORDER BY ord;'.format(videoname,splitkind,ord,ord+50)

    # print(sql2)
    cursor.execute(sql2)
    res = cursor.fetchall()

    cursor.close()
    conn.close()

    return res


def getTemporals_imgfilepath(splitkind='test'):

    sql1 = 'SELECT imgname,videoname,ord FROM ImgSets WHERE splitkind="{}" and imgkind="flow" ORDER BY RAND(),videoname,RAND(),ord LIMIT 1;'.format(splitkind)

    conn = ConnPool.connect()
    cursor = conn.cursor()
    cursor.execute(sql1)
    res = cursor.fetchone()

    videoname = res[1]
    ord = res[2]

    sql2 = 'SELECT imgpath,first_label FROM ImgSets WHERE videoname="{}" and splitkind="{}" and imgkind="flow" and ord>={} and ord<{} ORDER BY ord;'.format(videoname,splitkind,ord,ord+500)
    cursor.execute(sql2)
    res = cursor.fetchall()

    return res

def getMixs_imgfilepath(splitkind='test'):

    sql1 = 'SELECT imgpath,videoname,ord FROM ImgSets WHERE splitkind="{}" and imgkind="frame" ORDER BY RAND(),videoname,RAND(),ord LIMIT 1;'.format(splitkind)

    conn = ConnPool.connect()
    cursor = conn.cursor()
    cursor.execute(sql1)
    res = cursor.fetchone()

    picturename = res[0]
    videoname = res[1]
    ordl = res[2]-10
    ordr = res[2]+10


    sql2 = 'SELECT imgpath,first_label FROM ImgSets WHERE videoname="{}" and imgkind="flow" and ord>{} and ord <{} ORDER BY ord;'.format(videoname,ordl,ordr)

    cursor.execute(sql2)
    res = list(cursor.fetchall())

    return picturename,res

def getAllImgPath_From_Video():
    pass

def getVideoPictures_imgfilepathss():
    pass

