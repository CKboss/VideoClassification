WHERE = 'SERVER'

if WHERE == 'LOCAL':
    CODE_PATH = '/home/lab/Desktop/Development/VideoClassification/'
    TRAIN_SCRIPT = CODE_PATH + 'TFFusions/Train/train_config_yaml/'

elif WHERE == 'SERVER':

    CODE_PATH = '/mnt/md0/LSVC/Code/VideoClassification/'
    TRAIN_SCRIPT = CODE_PATH + 'TFFusions/Train/train_config_yaml/'

    DATA_PATH = '/mnt/md0/LSVC/'
    TRAIN_DATA = DATA_PATH + 'lsvc_train.txt'
    TEST_DATA = DATA_PATH + 'lsvc_test.txt'
    VAL_DATA = DATA_PATH + 'lsvc_val.txt'
    INDEX_DATA = DATA_PATH + 'lsvc_class_index.txt'

elif WHERE == 'SERVER202':

    CODE_PATH = '/datacenter/1/LSVC/Code/VideoClassification/'
    TRAIN_SCRIPT = CODE_PATH + 'TFFusions/Train/train_config_yaml/'

    DATA_PATH = '/datacenter/1/LSVC/'
    TRAIN_DATA = DATA_PATH + 'lsvc_train.txt'
    TEST_DATA = DATA_PATH + 'lsvc_test.txt'
    VAL_DATA = DATA_PATH + 'lsvc_val.txt'
    INDEX_DATA = DATA_PATH + 'lsvc_class_index.txt'
