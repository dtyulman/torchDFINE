import timeit

import torch
import numpy as np


class Timer:
    """http://preshing.com/20110924/timing-your-code-using-pythons-with-statement/"""
    def __init__(self, name='Timer', verbose=True):
        self.name = name
        self.verbose = verbose

    def __enter__(self):
        if self.verbose:
            print( 'Starting {}...'.format(self.name) )
        self.start = timeit.default_timer()
        return self

    def __exit__(self, *args):
        self.end = timeit.default_timer()
        self.elapsed = self.end - self.start
        if self.verbose:
            print( '{}: elapsed {} sec'.format(self.name, self.elapsed) )


def identity(x):
    return x


def int_sqrt(n, r=None, c=None):
    """Returns the two integers r,c such that their product is the closest one greater than n.
    Either r or c can be specified. Useful for making a r-by-c grid out of n items.
    """
    if r is None and c is None:
        r = int(np.ceil(np.sqrt(n)))
        c = int(np.round(np.sqrt(n)))
    elif r is None and c is not None:
        r = int(np.ceil(n/c))
    elif r is not None and c is None:
        c = int(np.ceil(n/r))
    assert r*c >= n #sanity check
    return r,c


class WrapperModule(torch.nn.Module):
    """Allows dynamically changing a child module, otherwise get a TypeError
    e.g. dfine.encoder = lambda x:x results in:
      `TypeError: cannot assign '__main__.<lambda>' as child module
      'encoder' (torch.nn.Module or None expected)`, but
    but dfine.encoder = WrapperModule(lambda x:x) is ok
    """
    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, *args, **kwargs):
        return self.f(*args, **kwargs)

    def __repr__(self):
        return repr(self.f)


class TempAttr:
    """If have object foo with foo.x = 'a', can do:
    with TempAttr(foo, 'x', 'b'):
        print(foo.x) #prints 'b'
    print(foo.x) #prints 'a'
    """
    def __init__(self, obj, attr, value):
        self.obj = obj
        self.attr = attr
        self.value = value

    def __enter__(self):
        self.original_value = getattr(self.obj, self.attr)
        setattr(self.obj, self.attr, self.value)

    def __exit__(self, exc_type, exc_val, exc_tb):
        setattr(self.obj, self.attr, self.original_value)



def tile_to_shape(tensor, target_shape):
    """
    Tile a tensor to match the desired target shape along all dimensions.
    If the target shape is not an integer multiple of the input shape,
    the tensor is truncated along each dimension.
    Pass -1 for a dimension in `target_shape` to leave it unchanged.

    Args:
        tensor (torch.Tensor): The input tensor to tile.
        target_shape (tuple): The desired shape of the output tensor, where -1 means "leave unchanged".

    Returns:
        torch.Tensor: The tiled and truncated tensor.
    """
    input_shape = tensor.shape
    if len(target_shape) != len(input_shape):
        raise ValueError("Target shape must have the same number of dimensions as the input tensor.")

    # Compute the number of repeats needed for each dimension
    repeats = [
        ((target_size + input_size - 1) // input_size if target_size != -1 else 1)
        for input_size, target_size in zip(input_shape, target_shape)
    ]

    # Tile the tensor
    tiled_tensor = tensor.repeat(*repeats)

    # Truncate to the target shape, preserving dimensions with -1
    slices = [
        slice(0, target_size if target_size != -1 else input_size)
        for input_size, target_size in zip(input_shape, target_shape)
    ]
    return tiled_tensor[tuple(slices)]



def linspace(start, end, steps, endpoint=True, **kwargs):
    if endpoint:
        return torch.linspace(start, end, steps, **kwargs)
    return torch.linspace(start, end * (steps-1)/steps, steps, **kwargs)


def verify_shape(tensor, shape):
    """
    Verify that tensor.shape[i] == shape[i]. If shape[i] is None, set it to tensor.shape[i]. If tensor is None, return shape
    """
    if tensor is not None:
        assert len(tensor.shape) == len(shape), 'Number of tensor dims different from specified shape dims'
        for i in range(len(shape)):
            if shape[i] is None:
                shape[i] = tensor.shape[i]
            else:
                assert shape[i] == tensor.shape[i], f'tensor.shape[{i}]={tensor.shape[i]} not equal shape[{i}]={shape[i]}'
    return shape


def verify_output_dim(fn, in_dim, out_dim):
    if fn is not None:
        fn_out_dim = fn(torch.zeros(in_dim)).shape[-1] #give dummy input to fn to get output shape
        if out_dim is None:
            out_dim = fn_out_dim
        else:
            assert out_dim == fn_out_dim, f'fn_out_dim={fn_out_dim} not equal out_dim={out_dim}'
    return out_dim


def approx_indexof(x, x_list):
    """
    Inputs:
        x: [v]. list, tuple, tensor, or array
        x_list: [b,v]. tensor

    Returns:
        x_nearest: [v], value from x_list closest to x
        idx: int, index of x_list with closest value to x
    """
    x = convert_to_tensor(x)
    idx = (x_list - x).abs().mean(dim=1).argmin()
    x_nearest = x_list[idx]
    return x_nearest, idx


def carry_to_device(data, device):
    '''
    Carries dict/list of torch Tensors/numpy arrays to desired device recursively

    Parameters:
    ------------
    - data: torch.Tensor/np.ndarray/dict/list: Dictionary/list of torch Tensors/numpy arrays or torch Tensor/numpy array to be carried to desired device
    - device: str, Device name to carry the torch Tensors/numpy arrays to

    Returns:
    ------------
    - data: torch.Tensor/dict/list: Dictionary/list of torch.Tensors or torch Tensor carried to desired device
    '''

    if torch.is_tensor(data):
        return data.to(device)

    elif isinstance(data, np.ndarray):
        return torch.tensor(data).to(device)

    elif isinstance(data, dict):
        for key in data.keys():
            data[key] = carry_to_device(data[key], device)
        return data

    elif isinstance(data, list):
        for i, d in enumerate(data):
            data[i] = carry_to_device(d, device)
        return data

    else:
        return data


def convert_to_tensor(x, convert_dtype='default'):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    elif isinstance(x, (list, tuple, float, int)):
        x = torch.tensor(x)
    elif not isinstance(x, torch.Tensor):
        raise ValueError('Invalid input')

    if not convert_dtype:
        pass
    elif convert_dtype == 'default':
        x = x.to(torch.get_default_dtype())
    else:
        x = x.to(convert_dtype)

    return x


def convert_to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    elif isinstance(x, torch.Tensor):
        return x.detach().numpy()
    elif isinstance(x, (list, tuple, float, int)):
        return np.array(x)
    raise ValueError('Invalid input')


def flatten_dict(dictionary, level=[]):
    '''
    Flattens nested dictionary by putting '.' between nested keys, reference: https://stackoverflow.com/questions/6037503/python-unflatten-dict

    Parameters:
    ------------
    - dictionary: dict, Nested dictionary to be flattened
    - level: list, List of strings for recursion, initialized by empty list

    Returns:
    ------------
    - tmp_dict: dict, Flattened dictionary
    '''

    tmp_dict = {}
    for key, val in dictionary.items():
        if isinstance(val, dict):
            tmp_dict.update(flatten_dict(val, level + [key]))
        else:
            tmp_dict['.'.join(level + [key])] = val
    return tmp_dict


def unflatten_dict(dictionary):
    '''
    Unflattens a flattened dictionary whose keys are joint string of nested keys separated by '.', reference: https://stackoverflow.com/questions/6037503/python-unflatten-dict

    Parameters:
    ------------
    - dictionary: dict, Flat dictionary to be unflattened

    Returns:
    ------------
    - resultDict: dict, Unflattened dictionary
    '''

    resultDict = dict()
    for key, value in dictionary.items():
        parts = key.split(".")
        d = resultDict
        for part in parts[:-1]:
            if part not in d:
                d[part] = dict()
            d = d[part]
        d[parts[-1]] = value
    return resultDict
