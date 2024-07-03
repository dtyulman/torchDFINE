import math
from copy import deepcopy

import torch
from torch import nn
from torch.utils.data import Dataset

from python_utils import verify_shape

#%%
class RNN(nn.Module):
    #https://www.proquest.com/docview/2668441219
    """
    Continuous-time RNN:
        tau dx/dt = -x + f(W @ x + b + Bs @ s + Bu @ u)
        y = Wy @ x + by
    where
        x_t: hidden neuron firing rate
        s_t: task input instructions
        u_t: control input (note, control input matrix Bu is fixed)
        y_t: task readout
    Discretized:
        x_{t+1} = (1-dt/tau)*x_t + dt/tau*f(W @ x_t + bx + Bs @ s_t + Bu @ u_t)
        y_t = Wy @ x_t + by
    """

    def __init__(self, dim_x, dim_y, dim_u, dim_s, dt=0.1, tau=1):
        super().__init__()

        #input parameters
        self.dim_s = dim_s
        self.Bs = nn.Parameter(torch.empty(dim_x, dim_s))

        self.dim_u = dim_u
        # self.Bu = torch.eye(dim_x, dim_u) #control input matrix B is fixed
        self.Bu = torch.randn(dim_x, dim_u) #control input matrix B is fixed

        #dynamics parameters
        self.dim_x = dim_x
        self.W = nn.Parameter(torch.empty(dim_x, dim_x))
        self.b = nn.Parameter(torch.empty(dim_x, 1))
        self.f = torch.tanh
        self.dt = dt
        self.tau = tau

        #readout parameters (feedthrough by default)
        if dim_y is None:
            self.dim_y = dim_x
            self.Wy = torch.eye(dim_x)
            self.by = 0
        else:
            self.dim_y = dim_y
            self.Wy = nn.Parameter(torch.empty(dim_y, dim_x))
            self.by = nn.Parameter(torch.empty(dim_y, 1))

        self._init_params()
        self.t = self.x = self.y = None


    def _init_params(self):
        #input parameters (note, control input matrix Bu is fixed)
        nn.init.eye_(self.Bs)

        #dynamics parameters
        gain = 4 #controls global dynamics, see https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.61.259
        nn.init.normal_(self.W, 0, gain/math.sqrt(self.dim_x))
        nn.init.zeros_(self.b)

        #readout parameters
        nn.init.normal_(self.Wy, 0, gain/math.sqrt(self.dim_y))
        nn.init.zeros_(self.by)


    def compute_observation(self, x, noise=False):
        """
        x: [b,x] or [b,t,x]
        """
        y = self.Wy @ x.unsqueeze(-1) + self.by #[y,x] @ [...,x,1] + [y,1] -> [...,y,1]
        if noise:
            raise NotImplementedError()
        return y.squeeze(-1)


    def compute_next_neurons(self, x, s=None, u=None):
        pre_x_next = self.W @ x.unsqueeze(-1) + self.b #[x,x] @ [b,x,1] + [x,1] -> [b,x,1]
        if s is not None:
            pre_x_next += self.Bs @ s.unsqueeze(-1) # += [x,s] @ [b,s,1] -> [b,x,1]
        if u is not None:
            pre_x_next += self.Bu @ u.unsqueeze(-1) # += [x,u] @ [b,u,1] -> [b,x,1]

        x_next = (1-self.dt/self.tau)*x + (self.dt/self.tau)*self.f(pre_x_next).squeeze(-1) #[b,x] + [b,x]
        return x_next


    def _update_state(self, s, u):
        """
        inputs:
            s: [b,s], task instruction input
            u: [b,u], control input
        returns:
            x_next: [b,x]
            y_next: [b,y]
        """
        self.x = self.compute_next_neurons(self.x, s, u)
        self.y = self.compute_observation(self.x)
        return self.x, self.y


    def _log_state(self):
        self.t += 1
        self.x_seq[:, self.t, :] = self.x
        self.y_seq[:, self.t ,:] = self.y


    def init_state(self, x0=None, s_seq=None, u_seq=None, num_seqs=None, num_steps=None):
        #extract/validate dimensions
        num_seqs, dim_x = verify_shape(x0, [num_seqs, self.dim_x])
        num_seqs, num_steps, dim_s = verify_shape(s_seq, [num_seqs, num_steps, self.dim_s])
        try:
            num_seqs, num_steps, dim_u = verify_shape(u_seq, [num_seqs, num_steps, self.dim_u])
        except AssertionError:
            num_seqs, _, dim_u = verify_shape(u_seq, [num_seqs, num_steps-1, self.dim_u])

        #set defaults if needed
        x0    = x0    if x0    is not None else torch.zeros(num_seqs, self.dim_x)
        s_seq = s_seq if s_seq is not None else torch.zeros(num_seqs, num_steps, self.dim_s)
        u_seq = u_seq if u_seq is not None else torch.zeros(num_seqs, num_steps, self.dim_u)

        #allocate memory for logging
        self.x_seq = torch.full((num_seqs, num_steps, self.dim_x), torch.nan) #[b,t,x]
        self.y_seq = torch.full((num_seqs, num_steps, self.dim_y), torch.nan) #[b,t,y]

        #initialize state
        self.t = -1
        self.x = x0
        self.y = self.compute_observation(x0)
        RNN._log_state(self)

        return x0, s_seq, u_seq, num_seqs, num_steps


    def step(self, s=None, u=None):
        next_state = self._update_state(s, u)
        self._log_state()
        return next_state


    def forward(self, x0=None, s_seq=None, u_seq=None, num_seqs=None, num_steps=None):
        """
        inputs: (zeroes by default)
            x0: [b,x], initial neuron firing rate
            s_seq: [b,t,s], task instruction sequence
            u_seq: [b,t,u], control input sequence
        returns:
            x_seq: [b,t,x], neural firing rate sequence
            y_seq: [b,t,y], observation sequence
        """
        #explicitly call class's init_state() to avoid subclasses from changing its behavior
        x, s_seq, u_seq, num_seqs, num_steps = RNN.init_state(self, x0=x0, s_seq=s_seq, u_seq=u_seq, num_seqs=num_seqs, num_steps=num_steps)
        for t in range(num_steps-1):
            x = self.compute_next_neurons(x=x, u=u_seq[:,t,:], s=s_seq[:,t,:]) #[b,x]
            self.x_seq[:,t+1,:] = x
        self.y_seq = self.compute_observation(self.x_seq)

        return self.x_seq, self.y_seq



class ReachRNN(RNN):
    def _update_state(self, s=None, u=None):
        self.x, self.v = super()._update_state(s, u)
        self.y = self.y + self.v*self.dt
        return self.x, self.v, self.y


    def _log_state(self):
        super()._log_state()
        self.v_seq[:, self.t ,:] = self.v


    def init_state(self, y0=None, **kwargs):
        # Initialize neurons, velocity
        x0, s_seq, u_seq, num_seqs, num_steps = super().init_state(**kwargs)

        # Initialize position
        #extract/validate dimensions
        num_seqs, dim_y = verify_shape(y0, [num_seqs, self.dim_y])

        #set defaults if needed
        y0 = y0 if y0 is not None else torch.zeros(num_seqs, self.dim_y)

        #allocate memory for logging
        self.v_seq = torch.full((num_seqs, num_steps, self.dim_y), torch.nan) #[b,t,y]

        #initialize state
        self.v_seq[:,0,:] = self.v = self.y #v is now the neuron readout (velocity)
        self.y_seq[:,0,:] = self.y = y0 #y is the position

        return x0, y0, s_seq, u_seq, num_seqs, num_steps


    def forward(self, x0=None, y0=None, s_seq=None, u_seq=None, num_seqs=None, num_steps=None):
        x0, y0, s_seq, u_seq, num_seqs, num_steps = self.init_state(x0=x0, y0=y0, s_seq=s_seq, u_seq=u_seq, num_seqs=num_seqs, num_steps=num_steps)
        self.x_seq, self.v_seq = super().forward(x0, s_seq, u_seq, num_seqs, num_steps)
        self.y_seq = self.velocity_to_position(self.v_seq, y0=y0)
        return self.x_seq, self.v_seq, self.y_seq


    def velocity_to_position(self, v_seq, y0=None):
        """ Integrate velocity starting from y0 to get position
        v_seq: [b,t,y], v_seq[:,t,i] is velocity at time t along dimension i
        y0: [b,y]
        """
        num_seqs, num_steps, dim_v = verify_shape(v_seq, [None, None, self.dim_y])
        num_seqs, dim_y = verify_shape(y0, [num_seqs, self.dim_y])

        y_seq = torch.empty(*v_seq.shape)
        y_seq[:,0,:] = torch.zeros(num_seqs, dim_y) if y0 is None else y0
        for t in range(1,num_steps):
            y_seq[:,t,:] = y_seq[:,t-1,:] + v_seq[:,t,:]*self.dt
        return y_seq
