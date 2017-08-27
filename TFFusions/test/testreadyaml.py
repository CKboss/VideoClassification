import yaml
import TFFusions.Config.Config as Config

filename = Config.TRAIN_SCRIPT + 'lstm-memory-cell1024.yaml'

with open(filename, 'r') as f:
    lines = f.readlines()
    txt = '\n'.join(lines)
    dict = yaml.load(txt)
