import tensorflow as tf
import TFFusions.WILLOW.willow_frame_level_models as frame_level_models

from TFFusions.Train.load_yaml_to_FLAG import LOAD_YAML_TO_FLAG,Get_GlobalFLAG

if __name__=='__main__':

    LOAD_YAML_TO_FLAG('./TFFusions/test/NetVLAD.yaml')

    FLAGS = Get_GlobalFLAG()

    netvlad = frame_level_models.NetVLADModelLF()

    x = tf.placeholder(dtype=tf.float32,shape=(32,64,1024+128))
    y = tf.placeholder(dtype=tf.int32,shape=(32,))
    z = netvlad.create_model(model_input=x,
                             vocab_size=500,
                             num_frames=y)
