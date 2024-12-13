'''
Copyright (c) 2023 University of Southern California
See full notice in LICENSE.md
Hamidreza Abbaspourazad*, Eray Erturk* and Maryam M. Shanechi
Shanechi Lab, University of Southern California
'''

import torch
import numpy as np


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


def convert_to_tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    elif isinstance(x, np.ndarray):
        return torch.from_numpy(x) # use np.ndarray as middle step so that function works with tf tensors as well
    elif isinstance(x, (list, tuple, float, int)):
        return torch.tensor(x)
    raise ValueError('Invalid input')


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
