from VideoClassification.utils.DBTools.MySqlTools.dbcore import ConnPool


# 借助MYSQL的文件提取接口

def getFrames_imgfilepath(splitkind='test'):
    '''
    :param splitkind: test / val / train
    :return:
    '''
    sql1 = 'SELECT imgname,videoname,ord FROM ImgSets WHERE splitkind="{}" and imgkind="frame" and MOD(ord,1000)=0 ORDER BY RAND(),videoname,ord LIMIT 1;'.format(
        splitkind)

    conn = ConnPool.connect()
    cursor = conn.cursor()
    cursor.execute(sql1)
    res = cursor.fetchone()

    # print(res)
    videoname = res[1]
    ord = res[2]

    sql2 = 'SELECT imgname FROM ImgSets WHERE videoname="{}" and splitkind="{}" and imgkind="frame" and ord>={} and ord<{} ORDER BY ord;'.format(
        videoname, splitkind, ord, ord + 100)

    cursor.execute(sql2)
    res = cursor.fetchall()

    cursor.close()
    conn.close()

    return res


def getTemporals_imgfilepath(splitkind='test'):
    sql1 = 'SELECT imgname,videoname,ord FROM ImgSets WHERE splitkind="{}" and imgkind="flow" and MOD(ord-1,1000000)=0 ORDER BY RAND(),videoname,ord LIMIT 1;'.format(
        splitkind)

    conn = ConnPool.connect()
    cursor = conn.cursor()
    cursor.execute(sql1)
    res = cursor.fetchone()

    videoname = res[1]
    ord = res[2]

    sql2 = 'SELECT imgname FROM ImgSets WHERE videoname="{}" and splitkind="{}" and imgkind="flow" and ord>={} and ord<{} ORDER BY ord;'.format(
        videoname, splitkind, ord, ord + 500000)
    cursor.execute(sql2)
    res = cursor.fetchall()

    return res


def getMixs_imgfilepath(splitkind='test'):
    sql1 = 'SELECT imgname,videoname,ord FROM ImgSets WHERE splitkind="{}" and imgkind="frame" and MOD(ord,1000)=0 ORDER BY RAND(),videoname,ord LIMIT 1;'.format(
        splitkind)

    conn = ConnPool.connect()
    cursor = conn.cursor()
    cursor.execute(sql1)
    res = cursor.fetchone()

    picturename = res[0]
    videoname = res[1]

    picturename = picturename.split('_')
    picturename[-1] = '10.jpg'
    newname = picturename[:-1]
    picturename = '_'.join(picturename)

    newname[1] = 'flow'
    newname = '_'.join(newname) + '_'

    sql2 = 'SELECT imgname FROM ImgSets WHERE videoname="{}" and imgkind="flow" and imgname like "{}%" ORDER BY ord;'.format(
        videoname, newname)

    cursor.execute(sql2)
    res = list(cursor.fetchall())

    return picturename, res


def getAllImgPath_From_Video():
    pass


def getVideoPictures_imgfilepathss():
    pass
