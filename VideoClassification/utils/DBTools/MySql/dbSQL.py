# build db set

'''

..../Class/....avi/xxx/orgin
..../Class/....avi/xxx/flow

DBSET:

TABLE 1 ImgSets:

id , splitkind [train/test/val] , imgfilepath [] , imgfilename [] \
imgkind [frame/optial] , video_name , first_label , ord

'''

CREATE_TABLE_ImgSets_SQL = 'CREATE TABLE ImgSets' \
                           '(' \
                           'ID INTEGER PRIMARY KEY AUTO_INCREMENT,' \
                           'splitkind VARCHAR(32),' \
                           'imgpath VARCHAR(512),' \
                           'imgname VARCHAR(512) UNIQUE ,' \
                           'imgkind VARCHAR(32),' \
                           'videoname VARCHAR(512),' \
                           'first_label INTEGER,' \
                           'ord INTEGER' \
                           ');'

'''

labels_tabel

id , videoname, labels

'''

CREATE_VIDEO_LABELS_SQL = 'CREATE TABLE VideoLabels' \
                          '(' \
                          'ID INTEGER PRIMARY KEY AUTO_INCREMENT,' \
                          'videoname VARCHAR(512) ,' \
                          'label INTEGER ,' \
                          'UNIQUE KEY (videoname,label)' \
                          ') ;'

'''
TABLE 2 VideoSet:

id , splitkind , videoname , videopath , label, imgnum

'''

CREATE_TABLE_VideoSets_SQL = 'CREATE TABLE VideoSets' \
                             '(' \
                             'ID INTEGER PRIMARY KEY AUTO_INCREMENT,' \
                             'splitkind VARCHAR(32),' \
                             'videoname VARCHAR(512) UNIQUE ,' \
                             'videopath VARCHAR(512),' \
                             'label INTEGER,' \
                             'imgnum INTEGER ' \
                             ');'


INSERT_NEW_IMAGE = 'INSERT INTO ImgSets (splitkind, imgpath, imgname, imgkind, videoname, first_label, ord) VALUES ' \
                   '(' \
                   '%s,%s,%s,%s,%s,%s,%s' \
                   ');'

INSERT_VIDEO_LABELS = 'INSERT INTO VideoLabels (videoname , label) VALUES ' \
                      '(%s,%s);'



