import os
import math
from copy import deepcopy

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from python_utils import verify_shape, linspace
from script_utils import Timer
from plot_utils import plot_parametric
from time_series_utils import generate_input_noise


class RNN(nn.Module):
    #https://www.proquest.com/docview/2668441219
    """
    Continuous-time RNN:
        tau dh/dt = -h + f(W @ h + b + Bs @ s + Bu @ u)
        z = Wz @ h + bz
    where
        h_t: hidden neuron firing rate
        s_t: task input instructions
        u_t: control input (control input matrix Bu is fixed)
        z_t: task readout
    Discretized:
        h_{t+1} = (1-dt/tau)*h_t + dt/tau*f(W @ h_t + b + Bs @ s_t + Bu @ u_t)
        z_t = Wz @ h_t + bz
    """

    def __init__(self, dim_h, dim_z=None, dim_s=None, dim_u=None, observe=None, dt=0.1, tau=1):
        super().__init__()

        #input parameters
        self.dim_s = dim_s or dim_h
        self.Bs = nn.Parameter(torch.empty(dim_h, dim_s))
        nn.init.eye_(self.Bs)

        self.dim_u = dim_u or dim_h
        self.Bu = nn.Parameter(torch.empty(dim_h, dim_u), requires_grad=False) #control input matrix B is fixed
        nn.init.eye_(self.Bu)

        #dynamics parameters
        self.dim_h = dim_h
        self.W = nn.Parameter(torch.empty(dim_h, dim_h))
        self.b = nn.Parameter(torch.empty(dim_h, 1))

        gain = 4 #controls global dynamics, see e.g. https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.61.259
        nn.init.normal_(self.W, 0, gain/math.sqrt(self.dim_h))
        nn.init.zeros_(self.b)

        self.f = torch.tanh
        self.dt = dt
        self.tau = tau
        assert 0 < self.dt/self.tau <= 1

        #readout parameters
        self.dim_z = dim_z or dim_h
        self.Wz = nn.Parameter(torch.empty(dim_z, dim_h))
        self.bz = nn.Parameter(torch.empty(dim_z, 1))

        nn.init.normal_(self.Wz, 0, 1/math.sqrt(self.dim_z))
        nn.init.zeros_(self.bz)

        # define what the observations are for the RNN: neural activity, outputs, or a function of either/both
        if observe is None or observe == 'neurons' or observe == 'h':
            self.obs_fn = lambda h,z: h
        elif observe == 'output' or observe == 'z':
            self.obs_fn = lambda h,z: z
        else:
            self.obs_fn = obs_fn
        self.dim_y = None #set during init


    def compute_next_neurons(self, h, s=None, u=None):
        pre_h_next = self.W @ h.unsqueeze(-1) + self.b #[h,h] @ [b,h,1] + [h,1] -> [b,h,1]
        if s is not None:
            pre_h_next += self.Bs @ s.unsqueeze(-1) # += [h,s] @ [b,s,1] -> [b,h,1]
        if u is not None:
            pre_h_next += self.Bu @ u.unsqueeze(-1) # += [h,u] @ [b,u,1] -> [b,h,1]

        h_next = (1-self.dt/self.tau)*h + (self.dt/self.tau)*self.f(pre_h_next).squeeze(-1) #[b,h] + [b,h]
        return h_next


    def compute_output(self, h, noise=False):
        """
        h: [b,h] or [b,t,h]
        """
        z = self.Wz @ h.unsqueeze(-1) + self.bz #[z,h] @ [...,h,1] + [z,1] -> [...,z,1]
        if noise:
            raise NotImplementedError()
        return z.squeeze(-1)


    def get_observation(self):
        return self.obs_fn(self.h, self.z)


    def _update_state(self, s, u):
        self.h = self.compute_next_neurons(self.h, s, u)
        self.z = self.compute_output(self.h)
        return self.h, self.z


    def _log_state(self):
        self.t += 1
        self.h_seq[:, self.t, :] = self.h
        self.z_seq[:, self.t ,:] = self.z


    def step(self, s=None, u=None):
        """
        inputs:
            s: [b,s], task instruction input
            u: [b,u], control input
        returns:
            h_next: [b,h]
            z_next: [b,z]
        """
        next_state = self._update_state(s, u)
        self._log_state()
        return next_state


    def init_state(self, h0=None, s_seq=None, u_seq=None, num_seqs=None, num_steps=None):
        #extract/validate dimensions
        num_seqs, dim_h = verify_shape(h0, [num_seqs, self.dim_h])
        num_seqs, num_steps, dim_s = verify_shape(s_seq, [num_seqs, num_steps, self.dim_s])
        num_seqs, num_steps, dim_u = verify_shape(u_seq, [num_seqs, num_steps, self.dim_u])

        #set defaults if needed
        h0    = h0    if h0    is not None else torch.zeros(num_seqs, self.dim_h)
        s_seq = s_seq if s_seq is not None else torch.zeros(num_seqs, num_steps, self.dim_s)
        u_seq = u_seq if u_seq is not None else torch.zeros(num_seqs, num_steps, self.dim_u)

        #allocate memory for logging
        self.h_seq = torch.full((num_seqs, num_steps, self.dim_h), torch.nan) #[b,t,h]
        self.z_seq = torch.full((num_seqs, num_steps, self.dim_z), torch.nan) #[b,t,z]

        #initialize state
        self.t = -1
        self.h = h0
        self.z = self.compute_output(self.h)
        RNN._log_state(self)

        #infer observation dimension
        self.dim_y = self.get_observation().shape[-1]

        return h0, s_seq, u_seq, num_seqs, num_steps


    def forward(self, h0=None, s_seq=None, u_seq=None, num_seqs=None, num_steps=None):
        """
        inputs: (zeroes by default)
            h0: [b,h], initial neuron firing rate
            s_seq: [b,t,s], task instruction sequence
            u_seq: [b,t,u], control input sequence
        returns:
            h_seq: [b,t,h], neural firing rate sequence
            z_seq: [b,t,z], observation sequence
        """
        #explicitly call class's init_state() to avoid subclasses from changing its behavior
        h, s_seq, u_seq, num_seqs, num_steps = RNN.init_state(self, h0=h0, s_seq=s_seq, u_seq=u_seq, num_seqs=num_seqs, num_steps=num_steps)
        for self.t in range(num_steps-1):
            h = self.compute_next_neurons(h=h, u=u_seq[:,self.t,:], s=s_seq[:,self.t,:]) #[b,h]
            self.h_seq[:,self.t+1,:] = h
        self.z_seq = self.compute_output(self.h_seq)

        return self.h_seq, self.z_seq



class ReachRNN(RNN):
    def _update_state(self, s=None, u=None):
        self.h, self.v = super()._update_state(s, u)
        self.z = self.z + self.v*self.dt
        return self.h, self.v, self.z


    def _log_state(self):
        super()._log_state()
        self.v_seq[:, self.t ,:] = self.v


    def init_state(self, z0=None, **kwargs):
        # Initialize neurons, velocity
        h0, s_seq, u_seq, num_seqs, num_steps = super().init_state(**kwargs)

        # Initialize position
        #extract/validate dimensions
        num_seqs, dim_z = verify_shape(z0, [num_seqs, self.dim_z])

        #set defaults if needed
        z0 = z0 if z0 is not None else torch.zeros(num_seqs, self.dim_z)

        #allocate memory for logging
        self.v_seq = torch.full((num_seqs, num_steps, self.dim_z), torch.nan) #[b,t,z]

        #initialize state
        self.v_seq[:,0,:] = self.v = self.z #v is now the neuron readout (velocity)
        self.z_seq[:,0,:] = self.z = z0 #z is the position

        return h0, z0, s_seq, u_seq, num_seqs, num_steps


    def forward(self, h0=None, z0=None, s_seq=None, u_seq=None, num_seqs=None, num_steps=None):
        h0, z0, s_seq, u_seq, num_seqs, num_steps = self.init_state(h0=h0, z0=z0, s_seq=s_seq, u_seq=u_seq, num_seqs=num_seqs, num_steps=num_steps)
        self.h_seq, self.v_seq = super().forward(h0, s_seq, u_seq, num_seqs, num_steps)
        self.z_seq = self.velocity_to_position(self.v_seq, z0=z0)
        return self.h_seq, self.v_seq, self.z_seq


    def velocity_to_position(self, v_seq, z0=None):
        """ Integrate velocity starting from z0 to get position
        v_seq: [b,t,z], v_seq[:,t,i] is velocity at time t along dimension i
        z0: [b,z]
        """
        num_seqs, num_steps, dim_v = verify_shape(v_seq, [None, None, self.dim_z])
        num_seqs, dim_z = verify_shape(z0, [num_seqs, self.dim_z])

        z_seq = torch.empty(*v_seq.shape)
        z_seq[:,0,:] = torch.zeros(num_seqs, dim_z) if z0 is None else z0
        for t in range(1,num_steps):
            z_seq[:,t,:] = z_seq[:,t-1,:] + v_seq[:,t,:]*self.dt
        return z_seq





###########
# Helpers #
###########


################
# RNN training #
################
def train_rnn(rnn, dataset, loss_fn=nn.MSELoss(), epochs=1000, batch_size=None, print_every=100):
    opt = torch.optim.Adam(rnn.parameters())
    loader = DataLoader(dataset, batch_size=batch_size or len(dataset))
    with Timer('Train RNN'):
        for epoch in range(epochs):
            for i, (s_input, z_target) in enumerate(loader):
                h_seq, z_seq = rnn(s_seq=s_input)
                loss = loss_fn(z_seq, z_target)
                opt.zero_grad()
                loss.backward()
                opt.step()
            if epoch % print_every == 0:
                print(f'epoch={epoch}, batch_loss={loss:.4f}')
    return rnn


def make_rnn(rnn_kwargs, dataset_kwargs, train_kwargs, seed=None, load=False, save=False):
    if load:
        rnn = torch.load(load)

    else:
        if seed is not None:
            torch.manual_seed(seed)

        # Generate dataset
        dataset = get_rnn_dataset(**dataset_kwargs)

        # Init and train RNN
        for i, dim in enumerate(['dim_s', 'dim_z']):
            if dim not in rnn_kwargs or rnn_kwargs[dim] is None:
                rnn_kwargs[dim] = dataset[0][i].shape[-1]
            assert rnn_kwargs[dim] == dataset[0][i].shape[-1]

        rnn = RNN(**rnn_kwargs)
        rnn = train_rnn(rnn, dataset, **train_kwargs)

        # Optionally save
        if save:
            with torch.no_grad():
                rnn.init_state(num_seqs=0, num_steps=1) #reset state variables
            save_path = os.path.join(os.getcwd(),'data', save)
            print(f'Saving to {save_path}')
            torch.save(rnn, save_path)

    return rnn, dataset


def make_perturbed_rnn(rnn, noise_std=0.01):
    with torch.no_grad():
        rnn.init_state(num_seqs=0, num_steps=1) #reset state variables
    perturbed_rnn = deepcopy(rnn)
    perturbed_rnn.W.data += torch.randn_like(perturbed_rnn.W.data)*noise_std
    return perturbed_rnn


#########################
# RNN training datasets #
#########################

def get_rnn_dataset(name, **kwargs):
    if name == 'reach':
        return ReachDataset(**kwargs)
    else:
        raise ValueError(f'Invalid dataset name: {name}')


class ReachDataset(Dataset):
    def __init__(self, num_steps=50, n_targets=8, spacing='radial'):
        self.num_steps = num_steps
        if spacing == 'radial':
            self.targets = torch.tensor([[torch.cos(x), torch.sin(x)] for x in linspace(0, 2*math.pi, n_targets, endpoint=False)]) #[b,z]
        elif spacing == 'uniform':
            self.targets = 2*torch.rand(n_targets, 2)-1 #[b,z]
        else:
            raise ValueError(f'Invalid spacing mode: {spacing}')


    def __getitem__(self, idx):
        try:
            input_seq = self.targets[idx].unsqueeze(0).expand(self.num_steps,-1) #[z]->[1,z]->[t,z]
        except:
            input_seq = self.targets[idx].unsqueeze(1).expand(-1,self.num_steps,-1) #[b,z]->[b,1,z]->[b,t,z]
        target_seq = input_seq.clone()
        return input_seq, target_seq


    def __len__(self):
        return len(self.targets)


    def plot_rnn_output(self, rnn, ax=None, title='Trained', label='Reach targets'):
        s_seq = self.targets.unsqueeze(1).expand(-1,self.num_steps,-1) #[b,z]->[b,1,z]->[b,t,z]
        h_seq, y_seq = rnn(s_seq=s_seq)
        fig, ax = plot_parametric(y_seq, cbar=False, ax=ax, varname='z', title=title)
        ax.scatter(*self.targets.T, c='k', label=label)
        if label:
            ax.legend()
        ax.axis('square')
        return fig, ax


    @torch.no_grad()
    def generate_DFINE_dataset(self, rnn, num_seqs, num_steps, **noise_params):
        u = generate_input_noise(rnn.dim_u, num_seqs, num_steps, **noise_params) #[b,t,u]
        s = self.targets.unsqueeze(1).tile(num_seqs//len(self), num_steps, 1) #[b,z]->[b,1,z]->[B,t,z]

        h, z = rnn(s_seq=s, u_seq=u) #[b,t,h], #[b,t,z]
        y = rnn.obs_fn(h, z) #[b,t,y]

        return h, z, y, u
