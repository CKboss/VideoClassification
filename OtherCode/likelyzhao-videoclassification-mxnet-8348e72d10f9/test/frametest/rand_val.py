import os
contents ={}
num_gpu = 4
rand_sum =100
f = open('trainvalfilelist.txt')
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
f.close()
f = open('lsvc_val.txt')
val_keys = []
for line in f.readlines():
    val_keys.append(line.split(',')[0])
f.close()

f = open('vallist_rand100.txt','w')
for key in contents:
    for val_key in val_keys:
        if val_key in key:
            print(key)
            if len(contents[key]) ==0:
                f.write(key)
                continue
            if len(contents[key]) <= rand_sum:
                f.write(key)
                for content in contents[key]:
                    f.write(content)
            if len(contents[key]) >rand_sum:
                f.write(val_key +' ' + str(rand_sum) + '\n')
                randidx =[]
                for i in range(rand_sum):
                    import random
                    randidx.append(random.randint(0,len(contents[key])-1))            
                for idx in randidx:
                    f.write(contents[key][idx])
f.close()
