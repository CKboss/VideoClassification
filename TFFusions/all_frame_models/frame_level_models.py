from TFFusions.all_frame_models.lstm_memory_model import LstmMemoryModel

def GetFrameModel(frame_model_name):
    if frame_model_name == 'LstmMemoryModel':
        return LstmMemoryModel
    else:
        raise NotImplementedError
