
WHERE='LOCAL'

if WHERE=='LOCAL':
    CODE_PATH = '/home/lab/Desktop/Development/VideoClassification/'
    TRAIN_SCRIPT = CODE_PATH+'TFFusions/train_scripts/train_config_yaml/'

elif WHERE=='SERVER':
    CODE_PATH = ''