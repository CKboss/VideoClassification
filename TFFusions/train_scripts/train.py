import TFFusions.Config.Config as Config
from TFFusions.train_scripts.load_yaml_to_FLAG import LOAD_YAML_TO_FLAG,Get_GlobalFLAG

from TFFusions.all_frame_models.frame_level_models import GetFrameModel

def main():

    train_config = Config.TRAIN_SCRIPT+'lstm-memory-cell1024.yaml'
    LOAD_YAML_TO_FLAG(train_config)

    FLAGS = Get_GlobalFLAG()

    model = GetFrameModel(FLAGS)

if __name__=='__main__':
    main()
