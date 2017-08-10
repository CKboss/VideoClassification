from VideoClassification.utils.DBTools.MySql.dbSQL import CREATE_TABLE_VideoSets_SQL,CREATE_VIDEO_LABELS_SQL,CREATE_TABLE_ImgSets_SQL,INSERT_NEW_IMAGE,INSERT_VIDEO_LABELS
from VideoClassification.utils.DBTools.MySql.dbcore import ConnPool


def CreateTabels():

    conn = ConnPool.connect()
    cursor = conn.cursor()

    cursor.execute(CREATE_TABLE_ImgSets_SQL)
    cursor.execute(CREATE_VIDEO_LABELS_SQL)
    cursor.execute(CREATE_TABLE_VideoSets_SQL)

    cursor.close()
    conn.close()

def chuliline(line: str) -> [str,str,str,str,list,int]:

    if len(line) == 0 :
        return

    line = line.split(' ')[0]
    item = line.split('\t')[0]

    if item[:4] == 'test':
        splitkind = 'test'
        imgname = item
        item = item[8:-4]
        items  = item.split('_')
        videoname = items[0]
        imgkind = items[1]
        labels = None

        if imgkind=='flow':
            ord = int(items[2])*1000000+\
                  int(items[3])*1000+\
                  int(items[4])
        elif imgkind=='frame':
            ord = int(items[-2])*1000 + int(item[-1])
        imagepath = None
        return splitkind,imagepath,imgname,imgkind,videoname,labels,ord
    elif item[:3] == 'val':
        splitkind = 'val'
    elif item[:5] == 'train':
        splitkind = 'train'

    imgname = item
    imagepath = None

    item = item[:-4]
    items  = item.split('_')

    print('items:',items)

    imgkind = items[1]

    if imgkind=='flow':
        ord = int(items[2])*1000000+ \
              int(items[3])*1000+ \
              int(items[4])
    elif imgkind=='frame':
        ord = int(items[-2])*1000 + int(item[-1])

    items = items[0].split('-')

    videoname = items[-1]

    labels = list(map(int,[ x for x in items[-2].split(',')]))

    return splitkind,imagepath,imgname,imgkind,videoname,labels,ord



def InsertIntoImageSets():

    filepath = '/home/itrc/Desktop/bucketlist_part.txt'
    # filepath = '/home/itrc/Desktop/test_line.txt'

    cnt = 0
    conn = ConnPool.connect()
    cursor = conn.cursor()

    with open(filepath,'r') as f:

        for line in f.readlines():
            splitkind,imagepath,imgname,imgkind,videoname,labels,ord = chuliline(line)
            cnt+=1
            if labels == None:
                templst = [splitkind,imagepath,imgname,imgkind,videoname,None,ord]
            else:
                templst = [splitkind,imagepath,imgname,imgkind,videoname,labels[0],ord]

            try:
                cursor.execute(INSERT_NEW_IMAGE,(templst))
            except Exception as E:
                print(E)

            if labels!=None:
                for l in labels:
                    try:
                        cursor.execute(INSERT_VIDEO_LABELS,(videoname,l))
                    except Exception as E:
                        print(E)

    cursor.close()
    conn.close()





if __name__=='__main__':

    # line1 = 'test--1-lsvc000005.avi_flow_0_0_1.jpg	10483	FlRYhnjGBGX_Hfrb4HxfQT4BXG6o	15023249189263833	image/jpeg	0'
    # line2 = 'train-159,23-lsvc032949.webm_frame_21_19.jpg	34217	FnnAbXN-ln6qXkCrH2FwpmRuIEBf	15023256195611639	image/jpeg	0'
    # line3 = 'val-159,23-lsvc032949.webm_frame_21_19.jpg	34217	FnnAbXN-ln6qXkCrH2FwpmRuIEBf	15023256195611639	image/jpeg	0'

    # chuliline(line3)
    InsertIntoImageSets()
