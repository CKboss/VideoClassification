import numpy as np
from numba import jit

from torch.nn.parameter import Parameter


@jit
def Accuracy(outputs: np.ndarray, targets: np.ndarray) -> float:
    '''
    :param outputs: 输出的logic值 np.array
    :param targets: 正确的标签
    :return: acc 正确比率
    '''
    n = len(targets)
    predicts = np.argmax(np.exp(outputs) / np.sum(np.exp(outputs)), axis=1)
    acc = np.sum(np.fabs(targets - predicts) < 1e-6) / n * 100
    return acc


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    res = [t.data[0] for t in res]
    return res


def try_to_load_state_dict(self, state_dict):
    """Copies parameters and buffers from :attr:`state_dict` into
    this module and its descendants. The keys of :attr:`state_dict` must
    exactly match the keys returned by this module's :func:`state_dict()`
    function.

    Arguments:
        state_dict (dict): A dict containing parameters and
            persistent buffers.
    """
    own_state = self.state_dict()

    for name, param in state_dict.items():

        if name not in own_state:
            print('try to load unexpected key "{}" in state_dict. just to skip it.'.format(name))
            continue

        if isinstance(param, Parameter):
            # backwards compatibility for serialized parameters
            param = param.data

        try:
            own_state[name].copy_(param)
        except RuntimeError:
            print('runtime error durring copy {}'.format(name))

    missing = set(own_state.keys()) - set(state_dict.keys())
    if len(missing) > 0:
        print('try to load missing keys in state_dict: "{}"'.format(missing))
