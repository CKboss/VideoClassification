import sqlite3

# build db set

'''

..../Class/....avi/xxx/orgin
..../Class/....avi/xxx/flow

DBSET:

TABLE 1 IMAGE:
id,filepath,label,kind(origin/flow)

TABLE 2 VIDEO:
id,filepath,short_screen_num,label,kind

'''

CREATE_TABLE_IMAGES_SQL = 'CREATE TABLE IMAGES ' \
                          '(' \
                          'ID INTEGER PRIMARY KEY autoincrement,' \
                          'filepath VARCHAR(200),' \
                          'label INTEGER,' \
                          'kind VARCHAR(10)' \
                          ');'


CREATE_TABLE_VIDEOS_SQL = 'CREATE TABLE VIDEOS' \
                          '(' \
                          'ID INTEGER PRIMARY KEY autoincrement,' \
                          'filepath VARCHAR(200),' \
                          'short_num INTEGER,' \
                          'label INTEGER,' \
                          'kind VARCHAR(10)' \
                          ');'

CREATE_TABLE_SQL = CREATE_TABLE_IMAGES_SQL+CREATE_TABLE_VIDEOS_SQL




