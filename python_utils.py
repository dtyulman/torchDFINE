'''
Copyright (c) 2023 University of Southern California
See full notice in LICENSE.md
Hamidreza Abbaspourazad*, Eray Erturk* and Maryam M. Shanechi
Shanechi Lab, University of Southern California
'''

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


def verify_shape(tensor, shape):
    """
    Verify that tensor.shape[i] == shape[i]. If shape[i] is None, set it to tensor.shape[i]
    """
    if tensor is not None:
        assert len(tensor.shape) == len(shape), 'Number of tensor dims different from specified shape dims'
        for i in range(len(shape)):
            if shape[i] is None:
                shape[i] = tensor.shape[i]
            else:
                assert shape[i] == tensor.shape[i], f'tensor.shape[{i}]={tensor.shape[i]} not equal shape[{i}]={shape[i]}'
    return shape


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
    '''
    Converts numpy.ndarray to torch.Tensor

    Parameters:
    ------------
    - x: np.ndarray, Numpy array to convert to torch.Tensor (if it's of type torch.Tensor already, it's returned without conversion)

    Returns:
    ------------
    - y: torch.Tensor, Converted tensor
    '''

    if isinstance(x, torch.Tensor):
        y = x
    elif isinstance(x, np.ndarray):
        y = torch.tensor(x) # use np.ndarray as middle step so that function works with tf tensors as well
    else:
        assert False, 'Only Numpy array can be converted to tensor'
    return y


def convert_to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().numpy()
    return x

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
