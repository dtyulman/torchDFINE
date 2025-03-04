'''
Copyright (c) 2023 University of Southern California
See full notice in LICENSE.md
Hamidreza Abbaspourazad*, Eray Erturk* and Maryam M. Shanechi
Shanechi Lab, University of Southern California
'''

import torch
from torch.utils.data import Dataset, IterableDataset

from data import RNN

class DFINEDataset(Dataset):
    '''
    Dataset class for DFINE.
    '''

    def __init__(self, y, u=None, behv=None, mask=None):
        '''
        Initializer for DFINEDataset. Note that this is a subclass of torch.utils.data.Dataset.

        Parameters:
        ------------
        - y: torch.Tensor, shape: (num_seq, num_steps, dim_y), High dimensional neural observations.
        - u: torch.Tensor, shape: (num_seq, num_steps, dim_u), Control input vectors.
        - behv: torch.Tensor, shape: (num_seq, num_steps, dim_behv), Behavior data. None by default.
        - mask: torch.Tensor, shape: (num_seq, num_steps, 1), Mask for manifold latent factors which shows whether
                                                              observations at each timestep exists (1) or are missing (0).
                                                              None by default.
        '''

        self.y = y

        # If control input is not provided, initialize it with 1-dimensional zero input
        if u is None:
            self.u = torch.zeros(*y.shape[:-1], 1)
        else:
            self.u = u

        # If behv is not provided, initialize it with zeros.
        if behv is None:
            self.behv = torch.zeros(y.shape[:-1]).unsqueeze(dim=-1)
        else:
            self.behv = behv

        # If mask is not provided, initialize it with ones.
        if mask is None:
            self.mask = torch.ones(y.shape[:-1]).unsqueeze(dim=-1)
        else:
            self.mask = mask


    def __len__(self):
        '''
        Returns the length of the dataset
        '''

        return self.y.shape[0]


    def __getitem__(self, idx):
        '''
        Returns a tuple of neural observations, behavior and mask segments
        '''
        return self.y[idx, :, :], self.u[idx, :, :], self.behv[idx, :, :], self.mask[idx, :, :]



class ControlledDFINEDataset(IterableDataset):
    def __init__(self, closed_loop, get_targets, num_steps):
       self.closed_loop = closed_loop
       self.get_targets = get_targets
       self.num_steps = num_steps


    def _update_dataset(self):
        self.y_target, self.plant_init, self.aux_inputs = self.get_targets()
        self.closed_loop.controller.K = self.closed_loop.controller.compute_lqr_gain(horizon=self.num_steps)
        self.closed_loop.run(y_target=self.y_target, plant_init=self.plant_init,
                             aux_inputs=self.aux_inputs, num_steps=self.num_steps)

        self.y = self.closed_loop.model.y #[b,t,y] true observation (logged by the model class for consistency)
        self.u = self.closed_loop.model.u #[b,t,u] input to plant, computed by model/controller
        self.y_hat = self.closed_loop.model.y_hat #[b,t,y] estimated observation


    def __len__(self):
        return self.get_targets.num_targets #[b]


    def __iter__(self):
        self._update_dataset()
        behv = torch.zeros(self.y.shape[:-1]).unsqueeze(dim=-1)
        mask = torch.ones(self.y.shape[:-1]).unsqueeze(dim=-1)
        yield from zip(self.y, self.u, behv, mask, self.y_target, self.y_hat)



class TargetGenerator():
    def __init__(self, num_targets, rnn, rnn_train_data):
        self.num_targets = num_targets
        self.rnn = rnn
        self.rnn_train_data = rnn_train_data
        self._verbose = True

    def __call__(self):
        #get num_targets random indices from rnn_train_data.targets
        idx = torch.randint(len(self.rnn_train_data), (self.num_targets,))
        z_target = self.rnn_train_data.targets[idx] #targets in z space

        #convert targets to observation space y (either h or z, depending on rnn.obs_fn)
        y_target, _, _ = RNN.make_y_target(self.rnn, z_target, h_target_mode='unperturbed',
                                           num_steps=self.rnn_train_data.num_steps, verbose=self._verbose)
        self._verbose = False

        #not used, but can be useful in the future if need to specify task input or specific init
        plant_init = None
        aux_inputs = None
        return y_target, plant_init, aux_inputs
