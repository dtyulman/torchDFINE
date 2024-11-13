'''
Copyright (c) 2023 University of Southern California
See full notice in LICENSE.md
Hamidreza Abbaspourazad*, Eray Erturk* and Maryam M. Shanechi
Shanechi Lab, University of Southern California
'''

from torch.utils.data import Dataset, IterableDataset
import torch


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
    def __init__(self, controller, R=1, xmin=-10, xmax=10, umin=-float('inf'), umax=float('inf'), num_steps=50, length=1):
       self.controller = controller
       self.R = R #TODO: set this dynamically? anneal?
       self.xmin = xmin
       self.xmax = xmax
       self.umin = umin
       self.umax = umax
       self.num_steps = num_steps

       self._length = length #used for compatibility with the epochs/batches framework
       self.outputs = None #store latest output of __next__ internally for analysis/debugging/etc


    def __iter__(self):
        self._num_seqs_generated = 0
        return self


    # def __len__(self):
    #     return self._length


    def __next__(self):
        if(self._num_seqs_generated >= self._length):
            raise StopIteration
        self._num_seqs_generated += 1

        # Pick a random initial and target point
        x0   = torch.rand(self.controller.model.dim_x)*(self.xmax-self.xmin) + self.xmin
        x_ss = torch.rand(self.controller.model.dim_x)*(self.xmax-self.xmin) + self.xmin

        # Run the contoller using the current model
        _, a_ss, y_ss = self.controller.generate_observation(x_ss)
        self.outputs = self.controller.run_control(y_ss, x_ss=x_ss, a_ss=a_ss, x0=x0, umin=self.umin, umax=self.umax, R=self.R, num_steps=self.num_steps)
        self.outputs['x_ss'] = x_ss
        self.outputs['a_ss'] = a_ss

        # Return the trajectory for training the next iteration of the model
        y, u = self.outputs['y'], self.outputs['u']
        behv = torch.zeros(y.shape[:-1]).unsqueeze(dim=-1)
        mask = torch.ones(y.shape[:-1]).unsqueeze(dim=-1)
        return y.squeeze(0), u.squeeze(0), behv.squeeze(0), mask.squeeze(0)
