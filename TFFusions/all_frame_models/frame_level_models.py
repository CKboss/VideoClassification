from TFFusions.all_frame_models.lstm_memory_model import LstmMemoryModel
from TFFusions.all_frame_models.lstm_model import LstmModel
from TFFusions.all_frame_models.lstm_cnn_deep_combine_chain_model import LstmCnnDeepCombineChainModel
from TFFusions.all_frame_models.lstm_attention_lstm_model import LstmAttentionLstmModel
from TFFusions.all_frame_models.lstm_attention_lstm_model_2 import LstmAttentionLstmModel2
from TFFusions.all_frame_models.gru_pooling_model import GruPoolingModel
from TFFusions.all_frame_models.bilstm_model import BiLstmModel
from TFFusions.WILLOW.willow_frame_level_models import NetVLADModelLF,NetVLADModelLF_VideoOnly

def GetFrameModel(frame_model_name):
    if frame_model_name == 'lstm_memory_model':
        return LstmMemoryModel
    elif frame_model_name == 'lstm_model':
        return LstmModel
    elif frame_model_name == 'LstmCnnDeepCombineChainModel':
        return LstmCnnDeepCombineChainModel
    elif frame_model_name == 'LstmAttentionLstmModel':
        return LstmAttentionLstmModel
    elif frame_model_name == 'LstmAttentionLstmModel2':
        return LstmAttentionLstmModel2
    elif frame_model_name == 'GruPoolingModel':
        return GruPoolingModel
    elif frame_model_name == 'bilstm':
        return BiLstmModel
    elif frame_model_name == 'NetVLADModelLF_VideoOnly':
        return NetVLADModelLF_VideoOnly
    elif frame_model_name == 'NetVLADModelLF':
        return NetVLADModelLF
    else:
        raise NotImplementedError
