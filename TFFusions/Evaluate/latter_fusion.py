import numpy as np

#############################
#    FUSION_NPZ_FILE_LIST   #
#############################

# each npz shold have correct_labels, predict_results, video_names

npz_file_list = [
    '/datacenter/1/LSVC/ExWorkSpace/Eval_lstm_memory_cell_1024_EX20/acc_1.binary.npz',
    '/datacenter/1/LSVC/ExWorkSpace/Eval_LstmAttentionModel_EX20/acc_1.binary.npz',
    '/datacenter/1/LSVC/downloads/NetVLAD_EX2_8000/acc_1.binary.npz',
    '/datacenter/1/LSVC/ExWorkSpace/Eval_GatedDbof_Video_EX1/acc_1.binary.npz',
]

model_num = len(npz_file_list)
weights = np.array([0.8, 0.8, 2.5, 2]) / model_num


#############################

class NPZ(object):
    cnt_1 = 0
    cnt_5 = 0
    cnt_10 = 0
    cnt = 0
    correct_labels = []
    predict_result = []
    video_names = []
    name2id = dict()


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

NPZs = []
common_video_name = None
common_video_num = None

# load files
def Chuli():

    global NPZs
    global common_video_name
    global common_video_num

    for file in npz_file_list:

        data = np.load(file)

        cnt_1,cnt_5,cnt_10,cnt = data['acc_1'],data['acc_5'],data['acc_10'],data['label_cnt']
        correct_labels = data['correct_labels']
        predict_result = data['predict_result']
        video_names = data['video_names']
        video_names = list(map(lambda x : x.decode(), video_names.tolist()))

        tmp = NPZ()
        tmp.cnt = cnt
        tmp.cnt_1 = cnt_1
        tmp.cnt_5 = cnt_5
        tmp.cnt_10 = cnt_10
        tmp.video_names = video_names
        tmp.correct_labels = []
        tmp.predict_result = []
        tmp.name2id = dict()

        for i,name in enumerate(tmp.video_names):
            goon = tmp.name2id.get(name,None)
            if goon is None:
                tmp.name2id[name] = i
                tmp.video_names.append(name)
                tmp.correct_labels.append(correct_labels[i])
                # do something
                p = predict_result[i]
                p -= np.max(p)
                p = (p - np.min(p)) / (np.max(p) - np.min(p))
                p = softmax(p)
                p = (p - np.min(p)) / (np.max(p) - np.min(p))
                tmp.predict_result.append(p)

        tmp.predict_result = np.concatenate(tmp.predict_result).reshape(-1,500)
        tmp.correct_labels = np.array(tmp.correct_labels)
        tmp.correct_labels = toOneHot(tmp.correct_labels)

        if common_video_name is None: common_video_name = set(tmp.video_names)
        else: common_video_name = common_video_name.intersection(set(tmp.video_names))

        NPZs.append(tmp)

    common_video_name = sorted(list(common_video_name))
    common_video_num = len(common_video_name)

    return NPZs

# load label
Labels = dict()
with open('/datacenter/1/LSVC/lsvc_class_index.txt','r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.replace('\n','')
        line = line.replace(' ','')
        ID,Name = line.split('\t')
        Labels[int(ID)-1] = Name

# Union It

Chuli()


pred = np.zeros((common_video_num,500))
label = np.zeros((common_video_num,500))


acc_1 = np.zeros(500)
acc_5 = np.zeros(500)
acc_10 = np.zeros(500)
cnt = np.zeros(500)

for id,video_name in enumerate(common_video_name):
    tmp_pred = None
    for i,npz in enumerate(NPZs):
        ID = npz.name2id[video_name]
        if tmp_pred is None: tmp_pred = npz.predict_result[ID]*weights[i]
        else: tmp_pred += npz.predict_result[ID]*weights[i]

    # tmp_pred /= model_num
    pred[id] = tmp_pred
    label[id] = npz.correct_labels[npz.name2id[video_name]]

    a = np.argmax(label[id])
    b = np.argsort(-pred[id])[:10]
    if a==b[0]:
        acc_1[a]+=1
    if a in b[:5]:
        acc_5[a]+=1
    if a in b:
        acc_10[a]+=1
    cnt[a]+=1


for zid in np.where(cnt==0)[0].tolist():
    cnt[zid] += 1
    print('add 1 to {}'.format(zid))

acc_1 = acc_1/cnt
acc_5 = acc_5/cnt
acc_10 = acc_10/cnt


# In[27]:
items = sorted(list(zip(acc_1.tolist(),list(range(500)))))
# In[28]:
for item in items:
    print('{:3}  {:30} {:.5f}    {:3.0f}'.format(item[1],Labels[item[1]],item[0]*100,cnt[item[1]]))

print('')
print('mean_ap: top_1 {:2f}% top_5 {:2f}% top_10 {:2f}%'.format(np.mean(acc_1)*100,np.mean(acc_5)*100,np.mean(acc_10)*100))
mAP = mean_ap(pred,label)
print('mAP:',mAP)

corr = 0
for i in range(len(label)):
    a = np.argmax(label[i])
    b = np.argmax(pred[i])
    corr += a==b

accuracy_all = corr/len(label)
print('accuracy_all:',accuracy_all)

def write_to_file(filename):
    with open(filename,'w') as f:
        for id,video_name in enumerate(common_video_name):
            part2 = str(pred[id].tolist())[1:-1]
            part2 = part2.replace(',','')
            line = '{}, {}\n'.format(video_name,part2)
            f.write(line)

# write_to_file('/home/qiu/t9.txt')

#
# for id,video_name in enumerate(common_video_name):
#     print(id)
#     print(video_name)
#     print(pred[id])
#     break
