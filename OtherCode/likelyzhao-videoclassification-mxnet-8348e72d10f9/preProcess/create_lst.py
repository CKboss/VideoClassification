import os
import numpy as np

ROOT = "/workspace/data/trainval"
LST_FRAME = "train_frame.lst"
LST_FLOW = "train_flow.lst"

TRAIN_PATH = "lsvc2017/lsvc_train.txt"
VAL_PATH = "lsvc2017/lsvc_val.txt"
label_map_train = {}
label_map_val ={}

def count():
    """
    count the number of the files
    :return: int,int
    """
    frame_train, flow_train,frame_val,flow_val = [], [], [], []
    label_frame_train, label_frame_val =[],[]
    label_flow_train, label_flow_val =[],[]
    for folder in os.listdir(ROOT):
        print(folder)
        if folder in label_map_train:
            if len(label_map_train[folder]) > 3:
                print(folder)
                continue
            filelist = os.listdir(os.path.join(ROOT,folder))
            filelist.sort()
#        print(len(filelist))
            for file in filelist:
                if 'frame' in file:
                    frame_train.append(os.path.join(folder, file))
                    label_frame_train.append(label_map_train[folder])
                if 'flow' in file:
                    flow_train.append(os.path.join(folder, file))
                    label_flow_train.append(label_map_train[folder])

        if folder in label_map_val:
             if len(label_map_val[folder]) > 3:
                print(folder)
                continue
             filelist = os.listdir(os.path.join(ROOT,folder))
             filelist.sort()
            # print(filelist)
             for file in filelist:
                if 'frame' in file:
                    frame_val.append(os.path.join(folder, file))
                    label_frame_val.append(label_map_val[folder])
                if 'flow' in file:
                    flow_val.append(os.path.join(folder, file))
                    label_flow_val.append(label_map_val[folder])
#        print(frame_account)

#        frame_account += len([file for file in os.listdir(os.path.join(ROOT, folder)) if file.__contains__("frame")])
 #       train_account += len([file for file in os.listdir(os.path.join(ROOT, folder)) if file.__contains__("flow")])

    return frame_train, flow_train, frame_val, flow_val,label_frame_train,label_flow_train,label_frame_val,label_flow_val


def init():
    with open(TRAIN_PATH, 'r') as f:
        for line in f:
            split_list = line.strip('\n').split(",")
            label_map_train[split_list[0]] = split_list[1:]
    with open(VAL_PATH,"r") as f:
        for line in f:
            split_list = line.strip('\n').split(",")
            label_map_val[split_list[0]] = split_list[1:] 

def rec_single_label(idx, labels, file_path):

    """
    
    :param idx: index
    :param labels: labels =[350,34]
    :param file_path: 
    :return: 
    """
    if len(labels) >= 1:
        labels = labels[0]
#    print(labels)
    return '{0}\t{1}\t{2}\n'.format(str(idx), labels,  file_path)


def rec(idx, labels, file_path):
    """
    
    :param idx: index
    :param labels: labels =[350,34]
    :param file_path: 
    :return: 
    """
    if len(labels) == 1:
        labels = [labels[0], "0", "0"]
    elif len(labels) == 2:
        labels.append("0")
    elif len(labels) > 3:
        assert False
#    print(labels)
    return '{0} \t {1} \t {2}\n'.format(str(idx), "\t".join(labels), file_path)


def random_frame(frame_account):
    """
    :param frame_account: frame account of file
    :return: 
    """
    arr = np.arange(frame_account)
    np.random.shuffle(arr)
    for num in arr.flat:
        yield num


def random_flow(flow_account):
    """
    :param flow_account: flow account number
    :return: yield a num
    """
    arr = np.arange(flow_account)
    np.random.shuffle(arr.flat)
    for num in arr:
        yield num


def build_lst():
    """
    
    :param typ: frame , flow
    :return: 
    """
    print(len(label_map_train))

    frame_train, flow_train,frame_val,flow_val,label_frame_train,label_flow_train,label_frame_val,label_flow_val = count()
    print(len(frame_train),len(flow_train),len(frame_val),len(flow_val))
    assert(len(frame_train) == len(label_frame_train))
    assert(len(flow_train) == len(label_flow_train))
    assert(len(frame_val) == len(label_frame_val))
    assert(len(flow_val) == len(label_flow_val))
 
    with open('train_frame_single.lst','w') as f:
        lines =[]
        arr = np.arange(len(frame_train))
        np.random.shuffle(arr.flat)
        for i in range(0,len(frame_train)):
            idx = arr[i]
            lines.append(rec_single_label(idx,label_frame_train[idx],frame_train[idx]))
        f.writelines(lines)

    with open('train_flow.lst','w') as f:
        lines =[]
#        arr = np.arange(len(flow_train))
#        np.random.shuffle(arr.flat)
        for i in range(0,len(flow_train)):
            idx = i
            lines.append(rec(idx,label_flow_train[idx],flow_train[idx]))
        f.writelines(lines)

    with open('val_frame_single.lst','w') as f:
        lines =[]
        arr = np.arange(len(frame_val))
        np.random.shuffle(arr.flat)
        for i in range(0,len(frame_val)):
            idx = arr[i]
            lines.append(rec_single_label(idx,label_frame_val[idx],frame_val[idx]))
        f.writelines(lines)

    with open('val_flow.lst','w') as f:
        lines =[]
#        arr = np.arange(len(flow_val))
#        np.random.shuffle(arr.flat)
        for i in range(0,len(flow_val)):
            idx = i
            lines.append(rec(idx,label_flow_val[idx],flow_val[idx]))
        f.writelines(lines)
   


def main():
    """
    main of the program
    :return: 
    """
    init()
    build_lst()


if __name__ == "__main__":
    main()
