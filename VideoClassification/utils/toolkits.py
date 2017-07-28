from torch.nn.parameter import Parameter


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
            print('try to load unexpected key "{}" in state_dict. just to skip it.' .format(name))
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

