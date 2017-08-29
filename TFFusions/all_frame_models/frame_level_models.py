from TFFusions.all_frame_models.lstm_memory_model import LstmMemoryModel
from TFFusions.all_frame_models.lstm_model import LstmModel
from TFFusions.all_frame_models.lstm_cnn_deep_combine_chain_model import LstmCnnDeepCombineChainModel
from TFFusions.all_frame_models.lstm_attention_lstm_model import LstmAttentionLstmModel
from TFFusions.all_frame_models.gru_pooling_model import GruPoolingModel

def GetFrameModel(frame_model_name):
    if frame_model_name == 'lstm_memory_model':
        return LstmMemoryModel
    elif frame_model_name == 'lstm_model':
        return LstmModel
    elif frame_model_name == 'LstmCnnDeepCombineChainModel':
        return LstmCnnDeepCombineChainModel
    elif frame_model_name == 'LstmAttentionLstmModel':
        return LstmAttentionLstmModel
    elif frame_model_name == 'GruPoolingModel':
        return GruPoolingModel
    else:
        raise NotImplementedError
