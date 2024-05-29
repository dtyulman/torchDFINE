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
        tau dx/dt = -x + f(W @ x_t + b + Bs @ s_t + Bu @ u_t)
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
        self.Bu = torch.eye(dim_x, dim_u) #control input matrix B is fixed

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


    def _init_params(self):
        #input parameters (note, control input matrix B is fixed)
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


    def step_neuron_dynamics(self, x, s=None, u=None):
        pre_x_next = self.W @ x.unsqueeze(-1) + self.b #[x,x] @ [b,x,1] + [x,1] -> [b,x,1]
        if s is not None:
            pre_x_next += self.Bs @ s.unsqueeze(-1) # += [x,s] @ [b,s,1] -> [b,x,1]
        if u is not None:
            pre_x_next += self.Bu @ u.unsqueeze(-1) # += [x,u] @ [b,u,1] -> [b,x,1]

        x_next = (1-self.dt/self.tau)*x + (self.dt/self.tau)*self.f(pre_x_next).squeeze(-1) #[b,x] + [b,x]
        return x_next


    def step(self, x, s=None, u=None):
        """
        inputs:
            x: [b,x], previous timestep neuron firing rate
            s: [b,s], task instruction input
            u: [b,u], control input
        returns:
            x_next: [b,x]
            y_next: [b,y]
        """
        x_next = self.step_neuron_dynamics(x, s, u)
        y_next = self.compute_observation(x_next)
        return x_next, y_next


    def init_state(self, x0=None, s_seq=None, u_seq=None, num_seqs=None, num_steps=None):
        num_seqs, dim_x = verify_shape(x0, [num_seqs, self.dim_x])
        num_seqs, num_steps, dim_s = verify_shape(s_seq, [num_seqs, num_steps, self.dim_s])
        num_seqs, num_steps, dim_u = verify_shape(u_seq, [num_seqs, num_steps, self.dim_u])

        x0    = x0    if x0    is not None else torch.zeros(num_seqs, self.dim_x)
        s_seq = s_seq if s_seq is not None else torch.zeros(num_seqs, num_steps, self.dim_s)
        u_seq = u_seq if u_seq is not None else torch.zeros(num_seqs, num_steps, self.dim_u)

        self.x_seq = torch.full((num_seqs, num_steps, self.dim_x), torch.nan) #[b,t,x]
        self.y_seq = torch.full((num_seqs, num_steps, self.dim_y), torch.nan) #[b,t,x]

        self.x_seq[:,0,:] = x0
        self.y_seq[:,0,:] = self.compute_observation(self.x_seq[:,0,:])

        return x0, s_seq, u_seq, num_seqs, num_steps


    def get_state(self, t):
        return self.x_seq[:,t,:]


    def set_state(self, x,y, t):
        self.x_seq[:,t,:] = x
        self.y_seq[:,t,:] = y


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
        #explicitly call class's init_state() to avoid subclasses from chaing its behavior
        x0, s_seq, u_seq, num_seqs, num_steps = RNN.init_state(self, x0=x0, s_seq=s_seq, u_seq=u_seq, num_seqs=num_seqs, num_steps=num_steps)

        #store x_seq as list and convert to tensor at the end, avoids in-place modification for proper gradient computation
        x_seq = [x0]
        for t in range(1,num_steps):
            x_t = self.step_neuron_dynamics(x=x_seq[t-1], u=u_seq[:,t-1,:], s=s_seq[:,t-1,:]) #[b,x]
            x_seq.append(x_t)
        self.x_seq = torch.stack(x_seq, dim=1) #[b,t,x]
        self.y_seq = self.compute_observation(self.x_seq)

        return self.x_seq, self.y_seq



class ReachRNN(RNN):
    def step(self, x, y, s=None, u=None):
        x_next, v_next = super().step(x, s, u)
        y_next = y + v_next*self.dt
        return x_next, v_next, y_next


    def init_state(self, y0=None, **kwargs):
        x0, s_seq, u_seq, num_seqs, num_steps = super().init_state(**kwargs)

        num_seqs, dim_y = verify_shape(y0, [num_seqs, self.dim_y])
        y0 = y0 if y0 is not None else torch.zeros(num_seqs, self.dim_y)

        self.v_seq = self.y_seq
        self.y_seq = torch.full((num_seqs, num_steps, self.dim_y), torch.nan) #[b,t,y]
        self.y_seq[:,0,:] = y0

        return x0, y0, s_seq, u_seq, num_seqs, num_steps


    def get_state(self, t):
        return self.x_seq[:,t,:], self.y_seq[:,t,:]


    def set_state(self, x,v,y, t):
        self.x_seq[:,t,:] = x
        self.v_seq[:,t,:] = v
        self.y_seq[:,t,:] = y


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
