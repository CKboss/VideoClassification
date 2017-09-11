import numpy as np

# file = '/datacenter/1/LSVC/downloads/accs_5.binary.npz'
# file = '/datacenter/1/LSVC/ExWorkSpace/Eval_LstmAttentionModel_EX20/acc_1.binary.npz'
# netvlad_3
# file = '/datacenter/1/LSVC/ExWorkSpace/Eval_NetVLAD_Video_EX3/acc_1.binary.npz'
# file = '/datacenter/1/LSVC/downloads/NetVLAD_EX2_8000/acc_1.binary.npz'
# file = '/datacenter/1/LSVC/ExWorkSpace/Eval_GatedDbof_Video_EX1/acc_1.binary.npz'


file_lst = ['/datacenter/1/LSVC/ExWorkSpace/Eval_lstm_memory_cell_1024_EX20/acc_1.binary.npz',
            '/datacenter/1/LSVC/ExWorkSpace/Eval_LstmAttentionModel_EX20/acc_1.binary.npz',
            '/datacenter/1/LSVC/downloads/NetVLAD_EX2_8000/acc_1.binary.npz',
            '/datacenter/1/LSVC/ExWorkSpace/Eval_GatedDbof_Video_EX1/acc_1.binary.npz',
            '/datacenter/1/LSVC/ExWorkSpace/Eval_GatedDbof_Video_EX3/acc_1.binary.npz',
            '/datacenter/1/LSVC/ExWorkSpace/Eval_BILSTM_EX4_78000/acc_1.binary.npz',
            ]

# file = '/datacenter/1/LSVC/ExWorkSpace/Eval_GatedDbof_Video_EX2/acc_1.binary.npz'

file = file_lst[-1]


data = np.load(file)

cnt_1,cnt_5,cnt_10,cnt = data['acc_1'],data['acc_5'],data['acc_10'],data['label_cnt']
correct_labels = data['correct_labels']
predict_result = data['predict_result']
video_names = data['video_names']
print('data.files:',data.files)

for zid in np.where(cnt==0)[0].tolist():
    cnt[zid] += 1
    print('add 1 to {}'.format(zid))

acc_1 = cnt_1/cnt
acc_5 = cnt_5/cnt
acc_10 = cnt_10/cnt

# In[127]:
video_name = list(map(lambda x : x.decode(), video_names.tolist()))

# In[128]:
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x))

def sigmoid(x):
    return 1/(1+np.exp(-x))

def tanh(x):
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

def normalize(v):
    norm=np.linalg.norm(v, ord=1)
    if norm==0:
        norm=np.finfo(v.dtype).eps
    return v/norm

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

def mean_ap(probs, labels):
    """
    Computes the mean average precision for all classes.
    :param probs: the predict probvalue batchsize x vocab_num
    :param labels: a one hot tensor which size is batchsize x vocab_num
    :return: a float num which is mean_ap
    """
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

def write_to_file(filename):
    with open(filename,'w') as f:
        for i in range(len(pred)):
            part2 = str(pred[i].tolist())[1:-1]
            part2 = part2.replace(',','')
            line = '{}, {}\n'.format(video_name[i],part2)
            f.write(line)

first_name = video_name[0]
bk = False

pred = []
label = []

for i,name in enumerate(video_name):
    if name == first_name:
        if bk == False:
            bk = True
        elif bk == True:
            break
    p = predict_result[i]
    p -= np.max(p)
    p = (p - np.min(p)) / (np.max(p) - np.min(p))
    p = softmax(p)
    # p = tanh(p)
    label.append(correct_labels[i])
    pred.append(p)

print('len:',len(pred))

pred = np.concatenate(pred).reshape(-1,500)
label = np.array(label)
label = toOneHot(label)

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

mAP = mean_ap(pred,label)
print('mAP:',mAP)

corr = 0
for i in range(len(label)):
    a = np.argmax(label[i])
    b = np.argmax(pred[i])
    corr += a==b
    # if a==0 and b==0:
    #     print(a,b)

accuracy_all = corr/len(label)
print('accuracy_all:',accuracy_all)

# write_to_file('/home/qiu/t5.txt')