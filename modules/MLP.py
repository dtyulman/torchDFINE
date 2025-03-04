'''
Copyright (c) 2023 University of Southern California
See full notice in LICENSE.md
Hamidreza Abbaspourazad*, Eray Erturk* and Maryam M. Shanechi
Shanechi Lab, University of Southern California
'''

import torch.nn as nn


class MLP(nn.Module):
    '''
    MLP Module for DFINE encoder and decoder in addition to the mapper to behavior for supervised DFINE.
    Encoder encodes the high-dimensional neural observations into low-dimensional manifold latent factors space
    and decoder decodes the manifold latent factors into high-dimensional neural observations.
    '''

    def __init__(self, **kwargs):
        '''
        Initializer for an Encoder/Decoder/Mapper object. Note that Encoder/Decoder/Mapper is a subclass of torch.nn.Module.

        Parameters
        ------------
        input_dim: int, Dimensionality of inputs to the MLP, default None
        output_dim: int, Dimensionality of outputs of the MLP , default None
        layer_list: list, List of number of neurons in each hidden layer, default None
        kernel_initializer_fn: torch.nn.init, Hidden layer weight initialization function, default nn.init.xavier_normal_
        activation_fn: torch.nn, Activation function of neurons, default nn.Tanh
        '''

        super(MLP, self).__init__()

        self.input_dim = kwargs.pop('input_dim', None)
        self.output_dim = kwargs.pop('output_dim', None)
        self.layer_list = kwargs.pop('layer_list', None)
        self.kernel_initializer_fn = kwargs.pop('kernel_initializer_fn', nn.init.xavier_normal_)
        self.activation_fn = kwargs.pop('activation_fn', nn.Tanh())

        # Create the hidden layers and initialize their weights based on desired initialization function
        bias = False #False for debugging computation graph
        ff = []
        current_dim = self.input_dim
        for i, dim in enumerate(self.layer_list):
            layer = nn.Linear(current_dim, dim, bias=bias)
            self.kernel_initializer_fn(layer.weight.data)
            current_dim = dim
            ff.append(layer)
            ff.append(self.activation_fn)

        out_layer = nn.Linear(current_dim, self.output_dim, bias=bias)
        self.kernel_initializer_fn(out_layer.weight.data)
        ff.append(out_layer)

        self.layers = nn.Sequential(*ff)


    def forward(self, inp):
        '''
        Forward pass function for MLP Module

        Parameters:
        ------------
        inp: torch.Tensor, shape: (num_seq * num_steps, input_dim), Flattened batch of inputs

        Returns:
        ------------
        out: torch.Tensor, shape: (num_seq * num_steps, output_dim),Flattened batch of outputs
        '''

        #although input to linear layer is shape (*,H) manually reshaping prevents
        #creating UnsafeView's for some reason...
        reshape_flag = False
        if len(inp.shape)>2:
            shape = inp.shape
            inp = inp.reshape(-1, inp.shape[-1])
            reshape_flag = True

        out = self.layers(inp)

        if reshape_flag:
            out = out.reshape(*shape[:-1], -1)

        return out
