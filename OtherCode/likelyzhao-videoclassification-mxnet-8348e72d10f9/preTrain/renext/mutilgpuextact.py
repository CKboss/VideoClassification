import os
contents ={}
num_gpu = 4
f = open('miss.txt')
lines = f.readlines()
i =0
while  i <  len(lines):
    key = lines[i]
    print(key)
    num = int(key.split(' ')[1])
    print(num)
    content = []
    for j in range(num):
        content.append(lines[i+1+j])
    i+=num
    i+=1
    contents[key] = content
idx =0
foutlist =[]
for i in range(num_gpu):
    foutlist.append(open('split_%d.txt'%i,'w'))
for key in contents:
    foutlist[idx%num_gpu].write(key)
    for content in contents[key]:
        foutlist[idx%num_gpu].write(content)
    idx+=1

for fpout in foutlist:
    fpout.close()
