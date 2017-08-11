from VideoClassification.utils.DBTools.MySqlTools.dbcore import ConnPool

conn = ConnPool.connect()

cur = conn.cursor()

from VideoClassification.utils.DBTools.MySqlTools.dbSQL import CREATE_TABLE_ImgSets_SQL,CREATE_TABLE_VideoSets_SQL,CREATE_VIDEO_LABELS_SQL

cur.execute(CREATE_TABLE_ImgSets_SQL)
cur.execute(CREATE_VIDEO_LABELS_SQL)
cur.execute(CREATE_TABLE_VideoSets_SQL)


