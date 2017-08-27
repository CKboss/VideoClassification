from TFFusions.all_video_models.moe_model import MoeModel


def GetVideoModel(video_model_name):
    if video_model_name == 'moe_model':
        return MoeModel
    else:
        raise NotImplementedError
