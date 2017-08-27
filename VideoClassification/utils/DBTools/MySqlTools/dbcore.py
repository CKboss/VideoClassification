import pymysql
import sqlalchemy.pool as DBPool

import VideoClassification.Config.Config as Config


# def getConnection():
#     conn = pymysql.connect(
#         host='',
#         user='root',
#         password='123456',
#         db='mmflow',
#         charset='utf8',
#         autocommit=True,
#     )
#     return conn


# def getConnection():
#     conn = pymysql.connect(
#         host='115.231.182.84',
#         user='root',
#         password='ava123',
#         db='mmflow',
#         charset='utf8',
#         autocommit=True,
#     )
#     return conn

def getConnection():
    conn = pymysql.connect(**Config.MySQL_Config)
    return conn


ConnPool = DBPool.QueuePool(getConnection, pool_size=1000, max_overflow=2000, timeout=25)

if __name__ == '__main__':
    conn = ConnPool.connect()
    cur = conn.cursor()
    ans = cur.execute('SELECT count(*) from ImgSets;')
    print(cur.fetchall())
