import sqlite3
import os.path

# build db set

'''

..../Class/....avi/xxx/orgin
..../Class/....avi/xxx/flow

DBSET:

TABLE 1 ImgSets:

id , splitkind [train/test/val] , imgfilepath [] , imgfilename [] \
imgkind [frame/optial] , video_name , label , ord

'''

CREATE_TABLE_ImgSets_SQL = 'CREATE TABLE ImgSets' \
                          '(' \
                          'ID INTEGER PRIMARY KEY autoincrement,' \
                          'splitkind VARCHAR(32),' \
                          'imgpath VARCHAR(1024),' \
                          'imgname VARCHAR(1024),' \
                          'imgkind VARCHAR(32),' \
                          'videoname VARCHAR(1024),' \
                          'label INTEGER,' \
                          'ord INTEGER' \
                          ');'

INSERT_NEW_IMAGE = 'INSERT INTO ImgSets (splitkind, imgpath, imgname, imgkind, videoname, label) VALUES ' \
                   '(' \
                   '?,?,?,?,?,?' \
                   ');'

'''
TABLE 2 VideoSet:

id , splitkind , videoname , videopath , label

'''

CREATE_TABLE_VideoSets_SQL = 'CREATE TABLE VideoSets' \
                             '(' \
                             'ID INTEGER PRIMARY KEY autoincrement,' \
                             'splitkind VARCHAR(32),' \
                             'videoname VARCHAR(1024),' \
                             'videopath VARCHAR(1024),' \
                             'label INTEGER' \
                             ');'

DB = './data/UCF101.db'

def CreateTable():
    os.path.exists('./data/UCF101.db')
    if os.path.exists('./data/UCF101.db') :
        print('DB alread create!')
        return
    with sqlite3.connect('./data/UCF101.db') as conn:
        cursor = conn.cursor()
        cursor.executescript(CREATE_TABLE_ImgSets_SQL+CREATE_TABLE_VideoSets_SQL)
        cursor.close()

def ExecutorSQL(sql):
    with sqlite3.connect(DB) as conn:
        cursor = conn.cursor()
        cursor.executescript(sql)
        cursor.close()

def InsertInToImages(items: list):
    with sqlite3.connect(DB) as conn:
        cursor = conn.cursor()
        cursor.executemany(INSERT_NEW_IMAGE,items)
        cursor.close()

CreateTable()
