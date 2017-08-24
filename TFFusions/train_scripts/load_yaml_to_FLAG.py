import yaml
import pprint

GLOBAL_FLAG = None

def LOAD_YAML_TO_FLAG(filename):

    global GLOBAL_FLAG

    class _flag(object): pass
    FLAG = _flag()
    with open(filename,'r') as f:
        txt = '\n'.join(f.readlines())
        data = yaml.load(txt)
        print('load config paramters: ',data['name'])
        pprint.pprint(data)
        for key in data.keys():
            FLAG.__setattr__(key,data[key])
    GLOBAL_FLAG = FLAG
    return FLAG

FLAGS = LOAD_YAML_TO_FLAG()

def Get_GlobalFLAG():
    global GLOBAL_FLAG
    return GLOBAL_FLAG

if __name__=='__main__':

    import TFFusions.Config.Config as Config
    filename = Config.TRAIN_SCRIPT+'lstm-memory-cell1024.yaml'
    F = LOAD_YAML_TO_FLAG(filename)
