import numpy as np
import sys
import argparse


parser = argparse.ArgumentParser(description='LSVC 2017 Computing Mean Average Precision')
parser.add_argument('--predicted', required=True, help='predicted scores')
parser.add_argument('--labels', required=True, default='lsvc_val.txt', help='ground truth labels')
if len(sys.argv)==1:
    parser.print_help()
    sys.exit(1)
args=parser.parse_args()




def mean_ap(probs, labels):
    """
    Computes the mean average precision for all classes.
    """
    mAP = np.zeros((probs.shape[1],1))
    for i in range(probs.shape[1]):
        iClass = probs[:,i]
        iY = labels[:,i]
        idx = np.argsort(-iClass)
        iY = iY[idx]
        count = 0
        ap = 0.0
        for j in range(iY.shape[0]):
            if iY[j] == 1:
                count = count + 1
                ap = ap + count/float(j+1)
            if count !=0:
                mAP[i] = ap/count
    return np.mean(mAP)


def get_score_matrix(scores):
    """
    Get the predicted scores.
    """
    score_mat = np.zeros((len(scores), 500))
    for i,p in enumerate(scores):
        vid_score = p.split(',')[1].strip().split(" ")
        score_mat[i,:] = np.asarray(vid_score, dtype=np.float32)
    return score_mat

def get_ground_truth(actual):
    """
    Get the ground truth labels.
    """
    label_mat = np.zeros((len(actual), 500))
    for i,p in enumerate(actual):
        vid_label = np.asarray(p.strip().split(',')[1:], dtype=np.int) - 1
        label_mat[i,vid_label] = 1
    return label_mat


def main(argv):

    predicted_file = open(args.predicted).read().splitlines()
    ground_truth_file = open(args.labels).read().splitlines()


    predicted_scores = get_score_matrix(predicted_file)
    ground_truth = get_ground_truth(ground_truth_file)

    mAP = mean_ap(predicted_scores, ground_truth)
    print(' The mean average precision is %.4f' % mAP)


if __name__ == "__main__":
    main(args)
