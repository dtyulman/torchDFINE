import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import control

from DFINE import DFINE
from data.SSM import NonlinearStateSpaceModel
from data.RNN import RNN, ReachRNN
from python_utils import verify_shape
from plot_utils import plot_vs_time


class LQGController():
    def __init__(self, plant, model, u_min=-float('inf'), u_max=float('inf')):
        """
        plant is an instance of NonlinearStateSpaceModel or RNN
        model is an instance of DFINE
        """
        self.model = model
        self.plant = plant

        assert self.plant.dim_y == self.model.dim_y, 'Observation dimensions must match in plant and model'

        #hard bounds on control inputs
        self.u_min = u_min
        self.u_max = u_max
        assert u_min < u_max


    def get_model_matrices(self):
        A = self.model.ldm.A
        B = self.model.ldm.B
        if 'RNN' in self.plant.__class__.__name__: #if isinstance(self.plant, RNN):
            Bs, Bu = self.model.ldm.B.split([self.plant.dim_s, self.plant.dim_u], dim=1) #[x,s], [x,u]
            B = Bu
        C = self.model.ldm.C
        return A, B, C


    def estimate_latents(self, y_seq, u_seq=None, ground_truth_latents=''):
        """
        inputs:
            y_seq: [b,t,y]: y_{0}  ... y_{t}    or [b,y]: y_{0}
            u_seq: [b,t,u]: u_{-1} ... u_{t-1}
            ground_truth_latents: 'a', 'x', 'ax', or ''. Only when plant is NonlinearStateSpaceModel

            #TODO: right now this runs the KF from 0 to t every time which defeats the purpose. Do this
                   recursively, try control.dlqe()
        """

        if len(y_seq.shape) == 2: #[b,y]
            y_seq = y_seq.unsqueeze(1) #[b,t,y]

        if ground_truth_latents:
            t = y_seq.shape[1] - 1 #only needed if using ground_truth_latents

        # Estimate model manifold latent state by inverting the observation via the encoder
        if 'a' in ground_truth_latents:
            a_hat_seq = self.plant.a_seq[:,:t+1,:]
        else:
            a_hat_seq = self.model.encoder(y_seq) #[b,t,a]: a_{0} ... a_{t}

        # Estimate dynamic latent from control inputs and history of manifold latents by Kalman filter
        if 'x' in ground_truth_latents:
            x_hat_seq = self.plant.x_seq[:,:t+1,:]
        else:
            x_hat_seq = self.model.ldm(a=a_hat_seq, u=u_seq)[1] #[b,t,x]: x_{0} ... x_{t}

        return x_hat_seq[:,-1,:], a_hat_seq[:,-1,:] #[b,x],[b,a]


    def verify_latents(self, x):
        """Check the observation to which the estimated latents correspond, via x->a->y"""
        _, _, C = self.get_model_matrices() #[a,x]
        a = (C @ x.unsqueeze(-1)).squeeze(-1) #[a,x]@[b,x,1]->[b,a]
        y = self.model.decoder(a) #[b,a]->[b,y]
        return a, y


    def compute_lqr_gain(self, Q=1, R=1, penalize_a=True):
        """
        Compute infinite-horizon LQR feedback gain matrix G
        Cost function: J = \\sum_t x^T Q x + u^T R u
        """
        A, B, C = self.get_model_matrices()

        if isinstance(Q, (float, int)):
            if penalize_a:  #effectively puts the penalty on a(t) with Q=I because a = C @ x
                Q = Q * (C.T @ C) #[x,a]@[a,x]->[x,x]
            else:
                Q = Q * torch.eye(self.model.dim_x) #[x,x]

        if isinstance(R, (float, int)):
            R = R * torch.eye(self.plant.dim_u) #[u,u]

        _, _, G = control.dare(A, B, Q, R) #[x,x],[x,u],[x,x],[u,u]->[u,x]
        G = torch.tensor(G, dtype=torch.float32)
        return G


    def compute_steady_state_control(self, x_ss, bias=0):
        """Compute steady-state control input"""
        A, B, _ = self.get_model_matrices() #[x,x],[x,u]
        I = torch.eye(self.model.dim_x) #[x,x]
        u_ss = torch.linalg.pinv(B) @ ((I-A)@x_ss.unsqueeze(-1) - bias) #[u,x] @ ([x,x]@[b,x,1] - [b,x,1]) -> [b,u,1]

        assert (self.u_min <= u_ss).all() and (u_ss <= self.u_max).all(), 'Steady state control input outside of min/max bounds'
        return u_ss.squeeze(-1) #[b,u]


    def init_model(self, y0, num_steps, u_prior=None, ground_truth_latents=''):
        """
        y0: [b,y]
        """
        num_seqs = y0.shape[0]

        self.x_hat = torch.full((num_seqs, num_steps, self.model.dim_x), torch.nan) #[b,t,x] dynamic latent via Kalman filter
        self.a_hat = torch.full((num_seqs, num_steps, self.model.dim_a), torch.nan) #[b,t,a] manifold latent via encoder
        self.a_hat_fwd = torch.full((num_seqs, num_steps, self.model.dim_a), torch.nan) #[b,t,a] manifold latent via C matrix
        self.y_hat = torch.full((num_seqs, num_steps, self.model.dim_y), torch.nan) #[b,t,y] observation estimate via decoder
        self.u     = torch.full((num_seqs, num_steps, self.plant.dim_u), torch.nan) #[b,t,u] control input

        self.x_hat[:,0,:], self.a_hat[:,0,:] = self.estimate_latents(y0, u_prior, ground_truth_latents)
        self.a_hat_fwd[:,0,:], self.y_hat[:,0,:] = self.verify_latents(self.x_hat[:,0,:])


    def run_control(self, y_ss, plant_init={}, Q=1, R=1, num_steps=100, t_on=-1, t_off=float('inf'), ground_truth_latents=''):
        """
        inputs:
            y_ss: [b,y], target
            plant_init: {'var1':[b,z1], ..., 'varN':[b,zN]} for each dynamic plant variable z
            ground_truth_latents: 'a', 'x', or 'ax' or ''. Only when plant is NonlinearStateSpaceModel
        """
        num_seqs = y_ss.shape[0]

        # Compute control constants
        if 'RNN' in self.plant.__class__.__name__: #if isinstance(self.plant, RNN): #u = [y_ss; u_lqr]
            Bs, _ = self.model.ldm.B.split([self.plant.dim_s, self.plant.dim_u], dim=1)
            bias = Bs @ y_ss.unsqueeze(-1) #[x,s] @ [b,s,1] -> [b,x,1]
            u_prior = torch.cat((y_ss.unsqueeze(1), #[b,s]->[b,1,s]
                                 torch.zeros(num_seqs,1,self.plant.dim_u)), #[b,1,u]
                                dim=-1) #[b,1,s+u]
        else:
            bias = 0
            u_prior = None

        G = self.compute_lqr_gain(Q, R) #[u,x]
        x_hat_ss, a_hat_ss = self.estimate_latents(y_ss, u_prior) #cannot use ground_truth_latents here, not defined #[b,y]->[b,x],[b,a]
        u_hat_ss = self.compute_steady_state_control(x_hat_ss, bias) #[b,x]->[b,u]

        # Initialization and logging
        self.plant.init_state(**plant_init, num_steps=num_steps, num_seqs=num_seqs)
        self.init_model(self.plant.y_seq[:,0,:], num_steps, u_prior=u_prior, ground_truth_latents=ground_truth_latents)

        # Run controller
        for t in range(num_steps-1):
            # Compute LQR control input
            if t_on <= t <= t_off:
                self.u[:,t,:] =  (-G @ (self.x_hat[:,t,:] - x_hat_ss).unsqueeze(-1)).squeeze(-1) + u_hat_ss #[u,x] @ [b,x,1] + [b,u] -> [b,u]
                self.u[:,t,:] = torch.clip(self.u[:,t,:], min=self.u_min, max=self.u_max) #optional hard bounds
            else:
                self.u[:,t,:] = 0

            # Step dynamics of plant with control input
            plant_t = self.plant.get_state(t)
            if not isinstance(plant_t, tuple):
                plant_t = (plant_t,)
            if 'RNN' in self.plant.__class__.__name__: #if isinstance(self.plant, RNN):
                plant_next = self.plant.step(*plant_t, s=y_ss, u=self.u[:,t,:])
            else:
                plant_next = self.plant.step(*plant_t, u=self.u[:,t,:])
            self.plant.set_state(*plant_next, t+1)

            # Estimate latents based on observation and input
            y_seq = self.plant.y_seq[:,:t+2,:] #[b,t,y]: y_{0}, ... y_{t+1}
            #assume input was zero for t<0, make u_seq=[u_{-1}, u_{0}, ..., u_{t}] with u_{-1}=0
            u_seq = self.u[:, :t+1, :] #[b,t,u]: u_{0}, ..., u_{t}
            u_seq = torch.cat((torch.zeros_like(u_seq[:,0:1,:]), u_seq), dim=1) #[b,t,u]: u_{-1},u_{0}, ..., u_{t}
            if 'RNN' in self.plant.__class__.__name__: #make u = [y_ss; u_lqr]
                s_seq = y_ss.unsqueeze(1).expand(-1,u_seq.shape[1],-1) #[b,y]->[b,t,y]
                u_seq = torch.cat((s_seq, u_seq), dim=-1) #[b,t,s+u]
            self.x_hat[:,t+1,:], self.a_hat[:,t+1,:] = self.estimate_latents(y_seq, u_seq, ground_truth_latents=ground_truth_latents)

            # Additional logging which observations correspond to the inferred latents
            self.a_hat_fwd[:,t+1,:], self.y_hat[:,t+1,:] = self.verify_latents(self.x_hat[:,t+1,:])
        #would like to do this instead but introduces numerical errors
        # self.a_hat_fwd, self.y_hat = self.verify_latents(self.x_hat)
        a_hat_ss_fwd, y_hat_ss = self.verify_latents(x_hat_ss) #[b,x]->[b,a],[b,y]

        return {'x_hat':self.x_hat,  'a_hat':self.a_hat,  'y_hat':self.y_hat, 'u':self.u, 'y':self.plant.y_seq,
                'a_hat_fwd':self.a_hat_fwd, 'a_hat_ss_fwd':a_hat_ss_fwd,
                'x_hat_ss':x_hat_ss, 'a_hat_ss':a_hat_ss, 'y_hat_ss':y_hat_ss, 'u_hat_ss':u_hat_ss, 'y_ss':y_ss,
                't_on':t_on, 't_off':t_off,
                'Q':Q, 'R':R}


    @staticmethod
    def plot_all(x_hat,    a_hat,    y_hat,    u,        y,
                 x_hat_ss, a_hat_ss, y_hat_ss, u_hat_ss, y_ss,
                 a_hat_fwd, a_hat_ss_fwd,
                 x_ss=None, a_ss=None, u_ss=None, x=None, a=None, #only when plant is NonlinearStateSpaceModel
                 t_on=-1, t_off=float('inf'),
                 Q=None, R=None,
                 seq_num=0, max_plot_rows=7):

        var_names = ['x', 'a', 'u', 'y']
        var_names_long = ['Dynamic latents', 'Manifold latents', 'Control inputs', 'Observations']

        max_dim = max([var.shape[-1] for var in (x_hat, a_hat, y_hat, u)])
        nrows = min(max_plot_rows, max_dim)
        fig, axs = plt.subplots(nrows, 4, sharex=True, sharey=False, figsize=(13, 1.6*nrows+1))

        def plot(seq, target, varname, color, label):
            B,T,N = seq.shape
            ax_num = var_names.index(varname)
            ax = np.expand_dims(axs[:N, ax_num], axis=1)
            plot_vs_time(seq[seq_num], target[seq_num], varname=varname, t_on=t_on, t_off=t_off,
                         ax=ax, max_N=nrows, legend=True, color=color, label=label)
            for a in axs[N:,ax_num]:
                a.axis('off')
            axs[0,ax_num].set_title(var_names_long[ax_num])
            return ax

        #inferred vars
        plot(x_hat, x_hat_ss, 'x', color='tab:blue', label='$\widehat{x}$')
        plot(a_hat, a_hat_ss, 'a', color='tab:blue', label='$\widehat{a}$')
        plot(u,     u_hat_ss, 'u', color='tab:blue', label='$u$')

        #decoded vars
        plot(y_hat, y_hat_ss, 'y', color='tab:green', label='$\widehat{y}$')
        plot(a_hat_fwd, a_hat_ss_fwd, 'a', color='tab:green', label='$\widetilde{a}$')

        #plant vars
        plot(y,     y_ss,     'y', color='tab:orange', label='$y$')
        if u_ss is not None:
            _u = torch.full_like(u, float('nan'))
            plot(_u, u_ss, 'u', color='tab:orange', label=None)
        if a is not None or a_ss is not None:
            a = torch.full_like(a_hat, float('nan')) if a is None else a
            a_ss = torch.full_like(a_hat_ss, float('nan')) if a_ss is None else a_ss
            plot(a, a_ss, 'a', color='tab:orange', label='$a$')
        if x is not None or x_ss is not None:
            x = torch.full_like(x_hat, float('nan')) if x is None else x
            x_ss = torch.full_like(x_hat_ss, float('nan')) if x_ss is None else x_ss
            plot(x, x_ss, 'x', color='tab:orange', label='$x$')

        fig.tight_layout()
        return fig, axs
