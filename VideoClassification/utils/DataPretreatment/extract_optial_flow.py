import os
from numba import jit
from multiprocessing import Pool

denseflow='/home/lab/Desktop/Development/dense_flow_fbf/denseFlow'

def JustDoIt(cmd):
    os.system(cmd)

@jit
def eachFile(filefrom, fileto):

    pool = Pool(processes=25)
    pathDir = os.listdir(filefrom)

    for allDir in pathDir:

        child = os.path.join('%s/%s' % (filefrom, allDir))
        childsavefile =  os.path.join('%s/%s' % (fileto, allDir))
        os.mkdir(childsavefile)
        avipathDir =  os.listdir(child)

        for subchild in avipathDir:

            avivideo = os.path.join('%s/%s' % (child, subchild))
            avivideosavefile = os.path.join('%s/%s' % (childsavefile, subchild))
            avivideosavefile = avivideosavefile[:-4]

            print('process {} ....'.format(avivideosavefile))

            os.mkdir(avivideosavefile)
            os.mkdir(avivideosavefile+'/flow_x')
            os.mkdir(avivideosavefile+'/flow_y')
            os.mkdir(avivideosavefile+'/image')

            cmd = '{} -f {} -x {}/flow_x/flow_x -y {}/flow_y/flow_y -i {}/image/image -b 20 -t 1 -d 0 -s 1'.format(denseflow,avivideo, avivideosavefile, avivideosavefile, avivideosavefile)
            pool.apply_async(JustDoIt,(cmd,))

    pool.close()
    pool.join()


if __name__ == '__main__':

    filePathI = '/home/lab/Desktop/Development/dense_flow_fbf/testfile-fbf/UCF101_images'
    filePathC = '/home/lab/Desktop/Development/dense_flow_fbf/testfile-fbf/UCF101'

    avipath = eachFile(filePathC, filePathI)
