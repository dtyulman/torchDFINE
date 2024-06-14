from modules.MLP import MLP
import torch.nn as nn
from nn import get_kernel_initializer_function, get_activation_function

class Autoencoder(nn.Module):

    def __init__(self,**kwargs):
        super().__init__()

        self.dim_y = kwargs.pop('dim_y', None)
        self.dim_a = kwargs.pop('dim_a', None)
        self.layer_list = kwargs.pop('layer_list', None)
        self.activation_str = kwargs.pop('activation_str', nn.Tanh)
        self.nn_kernel_initializer = kwargs.pop('nn_kernel_initializer', None)

        # Initialize the autoencoder
        self.encoder = self._get_MLP(input_dim=self.dim_y,
                                        output_dim=self.dim_a,
                                        layer_list=self.layer_list,
                                        activation_str=self.activation_str)

        self.decoder = self._get_MLP(input_dim=self.dim_a,
                                        output_dim=self.dim_y,
                                        layer_list=self.layer_list[::-1],
                                        activation_str=self.activation_str)


    def _get_MLP(self, input_dim, output_dim, layer_list, activation_str='tanh'):
        '''
        Creates an MLP object

        Parameters:
        ------------
        - input_dim: int, Dimensionality of the input to the MLP network
        - output_dim: int, Dimensionality of the output of the MLP network
        - layer_list: list, List of number of neurons in each hidden layer
        - activation_str: str, Activation function's name, 'tanh' by default

        Returns:
        ------------
        - mlp_network: an instance of MLP class with desired architecture
        '''

        activation_fn = get_activation_function(activation_str)
        kernel_initializer_fn = get_kernel_initializer_function(self.nn_kernel_initializer)

        mlp_network = MLP(input_dim=input_dim,
                          output_dim=output_dim,
                          layer_list=layer_list,
                          activation_fn=activation_fn,
                          kernel_initializer_fn=kernel_initializer_fn
                          )
        return mlp_network

    def forward(self, y):
        a_hat = self.encoder(y)
        y_hat = self.decoder(a_hat)
        return y_hat
