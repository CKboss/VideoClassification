import pymysql
import sqlalchemy.pool as DBPool


def getConnection():
    conn = pymysql.connect(
        host='localhost',
        user='root',
        password='123456',
        db='mmflow',
        charset='utf8',
        autocommit=True,
    )
    return conn


ConnPool = DBPool.QueuePool(getConnection, pool_size=1000, max_overflow=2000, timeout=25)

if __name__=='__main__':
    conn = ConnPool.connect()
    cur = conn.cursor()
    ans = cur.execute('')
    print(cur.fetchall())
