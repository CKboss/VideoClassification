import os
f = open('lsvc_val.txt')
fout = open("test_val_2.txt",'w')
for line in f.readlines():
    splits = line.split(',')
    fout.write(splits[0] + ' ' + splits[1].strip() + '\n')
f.close()
fout.close()
