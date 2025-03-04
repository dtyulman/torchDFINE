import os
import math
from copy import deepcopy

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

from python_utils import verify_shape, verify_output_dim, linspace, tile_to_shape, WrapperModule, identity
from script_utils import Timer
from plot_utils import plot_parametric
from time_series_utils import generate_input_noise, compute_control_error
import data.SSM as SSM


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

    def __init__(self, dim_h, dim_z=None, dim_s=None, dim_u=None, f=torch.tanh, gain=1., train_bias=True,
                 control_task_input=False, obs_fn='h', init_h='zeros', dt=0.1, tau=1):
        super().__init__()

        #dynamics parameters
        self.init_h = init_h
        self.dim_h = dim_h
        self.W = nn.Parameter(torch.empty(self.dim_h, self.dim_h))
        nn.init.normal_(self.W, 0, gain/math.sqrt(self.dim_h)) #gain controls global dynamics, see e.g. https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.61.259

        self.b = nn.Parameter(torch.empty(self.dim_h, 1), requires_grad=train_bias)
        nn.init.zeros_(self.b)

        self.f = f
        self.dt = dt
        self.tau = tau
        assert 0 < self.dt/self.tau <= 1

        #input parameters
        self.dim_s = dim_s or dim_h
        self.Bs = nn.Parameter(torch.empty(dim_h, dim_s))
        nn.init.normal_(self.Bs, 0, 1/math.sqrt(self.dim_s))

        if control_task_input:
            #control input effectively goes through the task input channel
            #actual implementation just sets the control input matrix equal to the task input matrix
            self.dim_u = self.dim_s
            self.Bu = nn.Parameter(torch.empty(self.dim_h, self.dim_u), requires_grad=False)
            self.Bu.data = self.Bs.data
        else:
            self.dim_u = dim_u or dim_h
            self.Bu = nn.Parameter(torch.empty(self.dim_h, self.dim_u), requires_grad=False) #control input matrix B is fixed
            nn.init.eye_(self.Bu)

        #readout parameters
        self.dim_z = dim_z or dim_h
        self.Wz = nn.Parameter(torch.empty(self.dim_z, self.dim_h))
        nn.init.normal_(self.Wz, 0, 1/math.sqrt(self.dim_z))
        self.bz = nn.Parameter(torch.empty(self.dim_z, 1))
        nn.init.zeros_(self.bz)

        if not isinstance(obs_fn, dict):
            obs_fn = {'obs': obs_fn}
        self.obs_fn = ObservationFunction(self, **obs_fn)
        self.dim_y = self.obs_fn.dim_y


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
        if h0 is None:
            if self.init_h == 'zeros' or self.init_h == 'zeroes':
                h0 = torch.zeros(num_seqs, self.dim_h)
            elif self.init_h == 'rand_unif':
                h0 = 2*torch.rand(num_seqs, self.dim_h)-1
            else:
                h0 = self.init_h(num_seqs, self.dim_h)
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


##############
# Subclasses #
##############
class VelocityRNN(RNN):
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
class ObservationFunction(nn.Module):
    def __init__(self, rnn, obs='h', dim_y=None):
        super().__init__()

        self.Wr = 0
        self.obs = obs

        if self.obs == 'h': #neurons
            if dim_y is None or dim_y == rnn.dim_h:
                self.Wr = torch.eye(rnn.dim_h)
                self.dim_y = rnn.dim_h
            else:
                assert dim_y < rnn.dim_h
                assert rnn.dim_h % dim_y == 0, f'dim_h={rnn.dim_h} must be a multiple of dim_y={dim_y}' #TODO: generalize this
                self.dim_y = dim_y
                self.Wr = torch.repeat_interleave(torch.eye(self.dim_y), rnn.dim_h//self.dim_y, dim=1)
                self.Wr *= self.dim_y/rnn.dim_h

        elif self.obs == 'z': #output
            self.Wr = torch.eye(rnn.dim_z)
            self.dim_y = rnn.dim_z

        else:
             raise ValueError(f'Invalid input: obs={obs}')


    def forward(self, h, z):
        if self.obs == 'h':
            return (self.Wr @ h.unsqueeze(-1)).squeeze(-1)
        elif self.obs == 'z':
            return (self.Wr @ z.unsqueeze(-1)).squeeze(-1)
        else:
            raise ValueError(f'Invalid input: ObservationFunction.obs={self.obs}')


    def inv(self, y, verbose=True):
        try:
            h_or_z = (torch.linalg.inv(self.Wr) @ y.unsqueeze(-1)).squeeze(-1)
        except RuntimeError as e:
            h_or_z = (torch.linalg.pinv(self.Wr) @ y.unsqueeze(-1)).squeeze(-1)
            if verbose:
                print(e, f'{self.obs} estimates done through pinv(Wr)')
        return h_or_z



###############
# Linear mode #
###############

def effective_ldm_params(rnn, s=None, skip_lin_check=False):
    if not skip_lin_check:
        assert rnn.f.__name__ == 'identity'
    A = (1-rnn.dt/rnn.tau) * torch.eye(*rnn.W.shape) + rnn.dt/rnn.tau * rnn.W
    B = rnn.dt/rnn.tau * rnn.Bu
    C = rnn.obs_fn.Wr
    Ws = 0
    if s is not None:
        Ws = (rnn.Bs @ s.unsqueeze(-1)).squeeze(-1)
    b = rnn.dt/rnn.tau * (Ws + rnn.b)
    return A,B,C,b


def effective_ldm_properties(rnn, s=None, skip_lin_check=False, verbose=True):
    A,B,C,b = effective_ldm_params(rnn, s, skip_lin_check)
    return SSM.get_ldm_properties(A,B,C)


def make_dfine_from_rnn(rnn, dfine, s=None, skip_lin_check=False):
    A,B,C,b = effective_ldm_params(rnn, s, skip_lin_check)

    assert dfine.ldm.dim_x == rnn.dim_h
    assert dfine.ldm.dim_a == rnn.dim_y
    assert dfine.dim_y == rnn.dim_y
    assert dfine.ldm.dim_u == rnn.dim_u

    dfine = deepcopy(dfine)
    with torch.no_grad():
        dfine.ldm.A.data = A
        dfine.ldm.B.data = B
        dfine.ldm.C.data = C

        dfine.ldm.W_log_diag.data = torch.log(1e-6*torch.ones(dfine.dim_x))
        dfine.ldm.R_log_diag.data = torch.log(1e-6*torch.ones(dfine.dim_a))

        dfine.decoder = WrapperModule(identity)
        dfine.encoder = WrapperModule(identity)
    return dfine



################
# RNN training #
################
class MSELossVelocityPenalty(nn.MSELoss):
    def __init__(self, velocity_weight=1.0, mask=None):
        super().__init__()
        self.mse_loss = nn.MSELoss(reduction='none')  # per-element MSE loss for masking
        self.velocity_weight = velocity_weight
        self.default_mask = mask.float() if mask is not None else None

    def forward(self, outputs, targets, mask=None):
        # Use the provided mask or fall back to the default mask
        mask = mask.float() if mask is not None else self.default_mask

        # Compute the element-wise MSE
        elementwise_loss = self.mse_loss(outputs, targets)

        if mask is not None:
            # Apply the mask
            masked_loss = elementwise_loss * mask
            mse = masked_loss.sum() / mask.sum()
        else:
            # Standard unmasked MSE loss
            mse = elementwise_loss.mean()

        # Compute the velocity penalty
        velocity = outputs[:, 1:, :] - outputs[:, :-1, :]
        velocity_penalty = torch.mean(velocity ** 2)

        # Combine MSE and velocity penalty
        total_loss = mse + self.velocity_weight * velocity_penalty
        return total_loss


def train_rnn(rnn, dataset, loss_fn=nn.MSELoss(), epochs=1000, batch_size=None, print_every=None):
    if print_every is None:
        print_every = max(epochs // 10, 1)
    opt = torch.optim.Adam(rnn.parameters())
    loader = DataLoader(dataset, batch_size=batch_size or len(dataset))
    with Timer('Train RNN'):
        for epoch in range(epochs):
            for i, batch in enumerate(loader):
                s_input, z_target = batch
                h_seq, z_seq = rnn(s_seq=s_input)
                loss = loss_fn(z_seq, z_target)
                opt.zero_grad()
                loss.backward()
                opt.step()
            if epoch % print_every == 0:
                print(f'epoch={epoch}, batch_loss={loss:.4f}')
    return rnn


def make_rnn(rnn_kwargs, dataset_kwargs, train_kwargs, seed=None, load=False, save=False, perturb=False):
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

        if perturb:
            rnn = make_perturbed_rnn(rnn, noise_std=perturb)

        # Optionally save
        if save:
            with torch.no_grad():
                rnn.init_state(num_seqs=0, num_steps=1) #reset state variables
            save_path = os.path.join(os.getcwd(),'data', save)
            print(f'Saving to {save_path}')
            torch.save(rnn, save_path)

    return rnn, dataset


@torch.no_grad()
def make_perturbed_rnn(rnn, noise_std=0.01):
    rnn.init_state(num_seqs=0, num_steps=1) #reset state variables
    perturbed_rnn = deepcopy(rnn)
    perturbed_rnn.W.data += torch.randn_like(perturbed_rnn.W.data)*noise_std
    perturbed_rnn.original = rnn
    return perturbed_rnn


@torch.no_grad()
def make_rnn_from_dfine(dfine):
    A = dfine.ldm.A.detach().clone(),
    B = dfine.ldm.B.detach().clone(),
    C = dfine.ldm.C.detach().clone(),

    assert dfine.encoder.f.__name__ == dfine.encoder.f.__name__ == 'identity'

    dim_x, _ = verify_shape(A, [None, None])
    dim_x, dim_u = verify_shape(B, [dim_x, None])
    dim_y, dim_x = verify_shape(C, [None, dim_x])

    rnn = RNN(dim_h=dim_x, dim_s=dim_u, f=identity(), obs_fn={'obs':'h', 'dim_y':dim_y},
              control_task_input=True, dt=1, tau=1)
    rnn.Wh = A
    rnn.Bs = B
    rnn.obs_fn.Wr = C

    return rnn



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


    def plot_rnn_output(self, rnn, plot='z', h_dims=(0,3), n_targets=None):
        if hasattr(rnn, 'original'):
            if plot == 'z':
                fig, ax = plt.subplots(1,2, sharex='row', sharey='row', figsize=(10, 4))
                self._plot_rnn_output(rnn.original, ax=ax[0], n_targets=n_targets, legend=False)
                h0 = rnn.original.h_seq[:,0,:]
                self._plot_rnn_output(rnn, h0=h0, ax=ax[1], n_targets=n_targets, title='Perturbed', cbar=True)
                fig.set_size_inches(8.22, 4)
            elif plot == 'h':
                fig = plt.figure(figsize=(9, 4.5))
                ax = np.array([fig.add_subplot(121, projection='3d'), fig.add_subplot(122, projection='3d')])
                self._plot_rnn_output(rnn.original, plot='h', h_dims=h_dims, ax=ax[0], n_targets=n_targets, legend=False)

                h0 = rnn.original.h_seq[:,0,:]
                self._plot_rnn_output(rnn, h0=h0, plot='h', h_dims=h_dims, ax=ax[1], n_targets=n_targets, title='Perturbed')
                fig.tight_layout()
        else:
            if plot == 'z':
                fig, ax = self._plot_rnn_output(rnn, n_targets=n_targets)
            elif plot == 'h':
                fig, ax = self._plot_rnn_output(rnn, plot='h', h_dims=h_dims, n_targets=n_targets)

        return fig, ax

    def _plot_rnn_output(self, rnn, h0=None, ax=None, n_targets=None, plot='z', h_dims=(0,3), title='Trained', legend=True, cbar=False):
        s_seq, _ = self[:] if n_targets is None else self[:n_targets]
        h_seq, z_seq = rnn(h0=h0, s_seq=s_seq)
        if plot == 'z':
            fig, ax = plot_parametric(z_seq, cbar=cbar, ax=ax, varname='z', title=title)

            ax.scatter(*z_seq[:,0,:].T, c='k', label='Init', marker='o')
            ax.scatter(*s_seq[:,0,:].T, c='k', label='Target', marker='*')
        elif plot == 'h':
            h_dims = slice(*h_dims)
            fig, ax = plot_parametric(h_seq[:,:,h_dims], cbar=cbar, ax=ax, varname='h', title=title)

            ax.scatter(*h_seq[:,0,h_dims].T, c='k', label='Init', marker='o')
            h_target = (torch.linalg.pinv(rnn.Wz) @ s_seq[:,0,:].unsqueeze(-1)).squeeze(-1) #[h,z]@[b,z,1]->[b,h,1]->[b,h]
            ax.scatter(*h_target[:,h_dims].T, c='k', label='Target $(h^* = W^+z^*)$', marker='*')
        else:
            raise ValueError()

        if legend:
            ax.legend()
        ax.axis('square')
        return fig, ax


    @torch.no_grad()
    def generate_DFINE_dataset(self, rnn, num_seqs, num_steps, include_task_input=True, add_task_input_to_noise=False, **noise_params):
        u = generate_input_noise(rnn.dim_u, num_seqs, num_steps, **noise_params) #[b,t,u]

        if add_task_input_to_noise:
            _s = tile_to_shape(self.targets.unsqueeze(1), (num_seqs, num_steps, -1)) #[b,z]->[B,t,z]
            u = u + _s

        if include_task_input: #give task input to RNN while generating input/output data
            if rnn.Bs.storage().data_ptr() == rnn.Bu.storage().data_ptr():
                raise ValueError("Don't include_task_input if using control_task_input in RNN")
            s = tile_to_shape(self.targets.unsqueeze(1), (num_seqs, num_steps, -1)) #[b,z]->[B,t,z]
        else:
            s = None

        h, z = rnn(s_seq=s, u_seq=u)
        y = rnn.obs_fn(h, z) #[b,t,y]

        return h, z, y, u


###########
# Control #
###########
def make_y_target(rnn, z_target, h_target_mode='pinv', num_steps=None, verbose=True):
    h_target = None
    if rnn.obs_fn.obs == 'h':
        # Both of these are cheating, in principle need to infer this, or infer y_target directly
        if h_target_mode == 'unperturbed': # the setting of h that produces z_target output in unperturbed rnn.

            try: #if no original attribute, means did not perturb rnn, so use the rnn itself
                assert rnn.Wz == rnn.original.Wz, 'Only works if output matrix (_rnn.Wz) is not perturbed'
                _rnn = rnn.orignal
            except AttributeError as e:
                if verbose:
                    print(e, ', using RNN itself to generate target h instead')
                _rnn = rnn

            with torch.no_grad():
                s_seq = z_target.unsqueeze(1).expand(-1,num_steps,-1)
                _rnn(s_seq=s_seq)
            h_target = _rnn.h_seq[:,-1,:] #[b,h]

        elif h_target_mode == 'pinv':
            #one possible setting of h that produces z_target. Unique only if Wz is invertible (which it probably is NOT)
            #TODO: How to find ALL the possible settings?
            #TODO: ...of these, how to find the one that is (most easily) achievable by the controller?
            #TODO: ...or simply have the entire SET be the control target?
            h_target = (torch.linalg.pinv(rnn.Wz) @ z_target.unsqueeze(-1)).squeeze(-1) #[h,z]@[b,z,1]->[b,h,1]->[b,h]

        else:
            raise ValueError(f'Invalid h_target_mode: {h_target_mode}')

        # effective z_target induced by choice of h_target
        z_target = rnn.compute_output(h_target) #[z,h]@[b,h,1]->[b,z,1]->[b,z]

    y_target = rnn.obs_fn(h_target, z_target)
    return y_target, h_target, z_target


def get_outputs_from_observations(rnn, y, y_target=None):
    #TODO: move to RNN class?
    if rnn.obs_fn.obs == 'z':
        z = rnn.obs_fn.inv(y, verbose=False)
        z_target = rnn.obs_fn.inv(y_target) if y_target is not None else None

    elif rnn.obs_fn.obs == 'h':
        h = rnn.obs_fn.inv(y, verbose=False)
        h_target = rnn.obs_fn.inv(y_target) if y_target is not None else None

        z = rnn.compute_output(h)
        z_target = rnn.compute_output(h_target) if h_target is not None else None

    return z, z_target


def _plot_outputs_2d(z, z_target=None, ax=None, estimated=False, cbar=False, line='solid', title=''):
    if line is None:
        z = z[:,(0,-1),:]
        mode = 'scatter'
        cbar = False
    else:
        mode = f'line_{line}'
    fig, ax = plot_parametric(z, ax=ax, cbar=cbar, mode=mode,
                              varname='\\widehat{z}' if estimated else 'z',
                              title=('Estimated' if estimated else 'Controlled')+title)

    if z_target is not None:
        ax.scatter(*z_target.T, color='k',
                   marker='o' if line=='solid' else 'x',
                   label='$\\widehat{z}^*$' if estimated else '$z^*$')
        ax.legend()

    ax.axis('square')
    return fig, ax


def plot_outputs_2d(model, rnn, check_z_seq=True, overlaid=False, sharexy=True, line='solid'):
    z, z_target = get_outputs_from_observations(rnn, model.y, model.y_target)
    z_hat, z_hat_target = get_outputs_from_observations(rnn, model.y_hat, model.y_hat_target)

    if z_target is None: print('Warning: true z_target not known/not plotted, because y_target not given (probably specifying x_hat_target instead).')
    if check_z_seq: assert torch.allclose(rnn.z_seq, z) #TODO: this will fail if obs Wr is not invertible

    Wz = '' if rnn.obs_fn.obs == 'z' else 'W_z'
    Wr_inv = '' if (rnn.obs_fn.Wr == torch.eye(rnn.obs_fn.dim_y)).all() else 'W_r^{-1}'
    eqn = f" $z = {Wz}{Wr_inv} y$"
    est_eqn = f" $\\widehat{{z}} = {Wz}{Wr_inv} \\widehat{{y}}$"

    if overlaid:
        fig, ax =_plot_outputs_2d(z_hat, z_hat_target, title=est_eqn, estimated=True, cbar=True, line='dotted')
        _plot_outputs_2d(z, z_target, ax=ax, title=eqn)
    else:
        fig, ax = plt.subplots(1,2, sharex=sharexy, sharey=sharexy, figsize=(8.5,4))
        _plot_outputs_2d(z, z_target, ax=ax[0], title=eqn, line=line)
        _plot_outputs_2d(z_hat, z_hat_target, ax=ax[1], title=est_eqn, estimated=True, cbar=True, line=line)
        fig.tight_layout()
    return fig, ax


@torch.no_grad()
def compute_control_errors(plant, model):
    errors = {}

    z_hat, z_hat_target = get_outputs_from_observations(plant, model.y_hat, model.y_hat_target)
    errors['x_hat'] = compute_control_error(model.x_hat, model.x_hat_target).numpy()
    errors['y_hat'] = compute_control_error(model.y_hat, model.y_hat_target).numpy()
    errors['z_hat'] = compute_control_error(z_hat, z_hat_target).numpy()

    if model.y_target is not None:
        errors['y'] = compute_control_error(model.y, model.y_target).numpy()
        z, z_target = get_outputs_from_observations(plant, model.y, model.y_target)
        errors['z'] = compute_control_error(z, z_target).numpy()

    return errors
