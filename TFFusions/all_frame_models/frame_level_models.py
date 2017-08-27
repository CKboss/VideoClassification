from TFFusions.all_frame_models.lstm_memory_model import LstmMemoryModel
from TFFusions.all_frame_models.lstm_model import LstmModel
from TFFusions.all_frame_models.lstm_cnn_deep_combine_chain_model import LstmCnnDeepCombineChainModel

def GetFrameModel(frame_model_name):
    if frame_model_name == 'lstm_memory_model':
        return LstmMemoryModel
    elif frame_model_name == 'lstm_model':
        return LstmModel
    elif frame_model_name == 'LstmCnnDeepCombineChainModel':
        return LstmCnnDeepCombineChainModel
    else:
        raise NotImplementedError
