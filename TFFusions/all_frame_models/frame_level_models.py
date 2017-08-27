from TFFusions.all_frame_models.lstm_memory_model import LstmMemoryModel
from TFFusions.all_frame_models.lstm_model import LstmModel


def GetFrameModel(frame_model_name):
    if frame_model_name == 'lstm_memory_model':
        return LstmMemoryModel
    if frame_model_name == 'lstm_model':
        return LstmModel
    else:
        raise NotImplementedError
