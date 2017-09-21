#coding:utf-8

import os
import os.path

def listfile(rootdir,suffix='',isDirlist= False):
        filelist = []
        Dirlist = []	 
	for parent,dirnames,filenames in os.walk(rootdir):    #三个参数：分别返回1.父目录 2.所有文件夹名字（不含路径） 3.所有文件名字##
            if isDirlist:
	        for dirname in dirnames:
                    Dirlist.append(dirname)
    	    for filename in filenames:                        #输出文件信息
                fullpath = os.path.join(parent,filename)
#                print(dirnames)
#        	print "the full name of the file is:" + os.path.join(parent,filename) #输出文件路径信息
                if suffix == '':
		    filelist.append(fullpath)
		else:
		    if os.path.splitext(fullpath)[1][1:] == suffix: 
		        filelist.append(fullpath)
	return filelist,Dirlist


if __name__ == '__main__':
	print listfile('.')
     	print listfile('.','py') 
     	print listfile('.','py',True) 

