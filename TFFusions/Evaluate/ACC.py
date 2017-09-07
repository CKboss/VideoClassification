
# coding: utf-8

# In[125]:

import numpy as np


# In[126]:

data = np.load('/datacenter/1/LSVC/downloads/accs_4.binary.npz')
cnt_1,cnt_5,cnt_10,cnt = data['acc_1'],data['acc_5'],data['acc_10'],data['label_cnt']
correct_labels = data['correct_labels']
predict_result = data['predict_result']
video_names = data['video_names']
data.files


# In[127]:

video_name = list(map(lambda x : x.decode(), video_names.tolist()))


# In[128]:

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x))


# In[114]:

# predict_result = softmax(predict_result)


# In[115]:

first_name = video_name[0]
bk = False

lines = []
for i,name in enumerate(video_name):
    if name == first_name:
        if bk == False:
            bk = True
        elif bk == True:
            break
    predict_result[i] = (predict_result[i] - np.mean(predict_result[i])) / np.var(predict_result[i])
    pred = softmax(predict_result[i])
    pred = str(pred.tolist())[1:-1]
    pred = pred.replace(',','')
    line = '{}, {}'.format(name,pred)
    lines.append(line)
print('len:',len(lines))
lines = '\n'.join(lines)

# In[124]:

p = predict_result[i]

def normalize(v):
    norm=np.linalg.norm(v, ord=1)
    if norm==0:
        norm=np.finfo(v.dtype).eps
    return v/norm

# In[116]:

with open('/tmp/t3.txt','w') as f:
    f.write(lines)


# In[36]:

np.where(cnt==0)


# In[37]:

cnt[133]+=1
cnt[345]+=1
cnt[472]+=1


# In[6]:

acc_1 = cnt_1/cnt
acc_5 = cnt_5/cnt
acc_10 = cnt_10/cnt


# In[7]:

# load label
Labels = dict()
with open('/datacenter/1/LSVC/lsvc_class_index.txt','r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.replace('\n','')
        line = line.replace(' ','')
        ID,Name = line.split('\t')
        Labels[int(ID)-1] = Name


# In[27]:

items = sorted(list(zip(acc_1.tolist(),list(range(500)))))


# In[28]:

for item in items:
    print('{:3}  {:30} {:.5f}    {:3.0f}'.format(item[1],Labels[item[1]],item[0]*100,cnt[item[1]]))


# In[12]:

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def toOneHot(x, vocab=500):
    '''
    :param x: a ndarray (batch)
    :param vocab: classes num
    :return: a one_hot array (batch x vocab)
    '''
    batchsize = x.shape[0]
    ret = np.zeros(shape=(batchsize, vocab))
    for i in range(batchsize):
        ret[i, x[i]] = 1
    return ret

def mean_ap(probs, labels, needSoftmax=True):
    """
    Computes the mean average precision for all classes.
    :param probs: the predict probvalue batchsize x vocab_num
    :param labels: a one hot tensor which size is batchsize x vocab_num
    :return: a float num which is mean_ap
    """

    if needSoftmax:
        probs = softmax(probs)

    mAP = np.zeros((probs.shape[1], 1))
    for i in range(probs.shape[1]):
        iClass = probs[:, i]
        iY = labels[:, i]
        idx = np.argsort(-iClass)
        iY = iY[idx]
        count = 0
        ap = 0.0
        for j in range(iY.shape[0]):
            if iY[j] == 1:
                count = count + 1
                ap = ap + count / float(j + 1)
            if count != 0:
                mAP[i] = ap / count
    return np.mean(mAP)


# In[15]:

correct_labels = toOneHot(correct_labels)


# In[29]:

mAP = mean_ap(predict_result,correct_labels,False)


# In[30]:

mAP


# In[31]:

mAP = mean_ap(predict_result,correct_labels)


# In[32]:

mAP

