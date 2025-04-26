from copy import deepcopy

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import torch
import control
from cvxopt import matrix, solvers
solvers.options['show_progress'] = False  # Suppress solver output

from data.SSM import NonlinearStateSpaceModel
from python_utils import convert_to_tensor, convert_to_numpy, verify_shape
from plot_utils import plot_vs_time, plot_eigvals


class ClosedLoopSystem():
    """
    class Plant():
        def init_state(self, **plant_init):
            pass
        def step(self, u, **aux_inputs):
            return state
        def get_observation(self):
            return y

    class Model():
        def init_model(self, y0, num_steps):
            pass
        def estimate_target(self, y_target):
            pass
        def estimate_state(self, y, u, full_state):
            pass
    """
    def __init__(self, plant, model, controller):
        self.model = model
        self.plant = plant
        self.controller = controller


    def run(self, num_steps, y_target=None, x_hat_target=None, t_on=0, t_off=float('inf'), plant_init=None, aux_inputs=None):

        # Defaults and convenience vars
        num_seqs, _ = verify_shape(y_target, [None, self.plant.dim_y])
        num_seqs, _ = verify_shape(x_hat_target, [num_seqs, self.model.dfine.dim_x])

        aux_inputs = aux_inputs or [{}]
        if len(aux_inputs) == 1:
            aux_inputs *= num_steps
        else:
            assert len(aux_inputs) == num_steps
        #sanity check
        for aux_input_t in aux_inputs:
            for val in aux_input_t.values():
                assert val.shape[0] == num_seqs


        # Initialization and logging
        #model target
        x_hat_target = self.model.estimate_target(y_target, x_hat_target)

        #plant
        plant_init = plant_init or {}
        if plant_init == 'x_hat_target':
            plant_init = {'x0': x_hat_target} #for NonlinearStateSpaceModel only
        self.plant.init_state(**plant_init, num_seqs=num_seqs, num_steps=num_steps)
        y0 = self.plant.get_observation()

        #model init
        x_hat = self.model.init_state(y0, num_steps)

        #controller
        self.model.u_hat_target = self.controller.init_target(x_hat_target)

        # Run controller
        for t in range(0, num_steps-1):
            assert t == self.plant.t == self.model.t #sanity check

            # Compute control input
            if t_on <= t <= t_off:
                assert t == self.controller.t+t_on #sanity check
                u_t = self.controller(x_hat) #[b,u]
            else:
                u_t = torch.zeros_like(self.model.u_hat_target)

            # Step dynamics of plant with control input
            full_state = self.plant.step(u=u_t.detach(), **aux_inputs[t]) #e.g. (x_{t+1}, a_{t+1}, y_{t+1}) for NL-SSM or (h_{t+1}, z_{t+1}) for RNN
            y = self.plant.get_observation() #y_{t+1} [b,y]

            # Estimate latents based on observation and input
            x_hat = self.model.estimate_state(y, u_t) #\hat{x}_{t+1} [b,x]

        return



class DFINE_Wrapper:
    def __init__(self, dfine, plant, ground_truth=''):
        self.dfine = dfine
        self.plant = plant

        assert ground_truth in ['a', 'x', 'ax', 'xa', '']
        self.ground_truth = ground_truth
        if ground_truth:
            assert isinstance(plant, NonlinearStateSpaceModel), 'Can only use ground truth latents if plant is NonlinearStateSpaceModel'


    def init_state(self, y0, num_steps):
        """
        y0: [b,y]
        """
        num_seqs = y0.shape[0]

        self.y         = torch.full((num_seqs, num_steps,   self.dfine.dim_y), torch.nan) #[b,t,y] actual observation
        self.a_hat     = torch.full((num_seqs, num_steps,   self.dfine.dim_a), torch.nan) #[b,t,a] manifold latent via encoder
        self.x_hat     = torch.full((num_seqs, num_steps,   self.dfine.dim_x), torch.nan) #[b,t,x] dynamic latent via Kalman filter
        self.a_hat_fwd = torch.full((num_seqs, num_steps,   self.dfine.dim_a), torch.nan) #[b,t,a] manifold latent via C matrix
        self.y_hat     = torch.full((num_seqs, num_steps,   self.dfine.dim_y), torch.nan) #[b,t,y] observation estimate via decoder
        self.u         = torch.full((num_seqs, num_steps-1, self.plant.dim_u), torch.nan) #[b,t-1,u] control input (one step shorter because Tth input never used)

        self.t = -1
        x_hat_0 = self.estimate_state(y0)
        return x_hat_0


    def estimate_target(self, y_target=None, x_hat_target=None):
        """
        y_target: [b,y]. If specified, infers x_hat_target using the model.
        x_hat_target: [b,x]. If specified, ignores y_target and uses this as the target instead
        """

        self.y_target = y_target

        # Estimate model manifold latent state by inverting the observation (target) via the encoder
        self.a_hat_target = self.dfine.encoder(y_target) if y_target is not None else None #[b,a]

        if x_hat_target is None:
            # Kalman update simplifies to C matrix inverse, assuming system has reached steady-state:
            #    x_ss = Ax_ss + Bu_ss + b + K(a_ss - C(Ax_ss + Bu_ss + b))
            # => x_ss = x_ss + K(a_ss - Cx_ss), since x_ss = Ax_ss + Bu_ss + b
            # => x_ss = Cinv @ a_ss

            #TODO: what if the system is NOT at steady state? How to get x_hat_target??
            C = self.dfine.ldm.C
            self.x_hat_target = (torch.linalg.pinv(C) @ self.a_hat_target.unsqueeze(-1)).squeeze(-1) #[x,a]@[b,a,1]->[b,x,1]->[b,x]
        else:
            assert len(x_hat_target.shape) == 2
            self.x_hat_target = x_hat_target #[b,x]

        # Estimate observation corresponding to latent target state estimate
        self.a_hat_fwd_target, self.y_hat_target = self.estimate_observation(self.x_hat_target)

        # Make sure shape is [b, dim_var]
        num_seqs = self.x_hat_target.shape[0]
        for var in [self.y_target, self.a_hat_target, self.x_hat_target, self.a_hat_fwd_target, self.y_hat_target]:
            if var is not None: assert len(var.shape) == 2 and var.shape[0] == num_seqs #sanity check

        return self.x_hat_target #[b,x]


    def estimate_state(self, y, u=None):
        """
        y: [b,y]
        u: [b,u]
        """
        #TODO: this re-implements some of the logic in DFINE.forward() so that it can include
        # the ground_truth option, try to reuse/modify that instead

        # Log the observation and input
        self.t += 1
        self.y[:, self.t, :] = y
        if self.t > 0:
            self.u[:, self.t-1, :] = u

        # Estimate model manifold latent state by inverting the observation via the encoder
        if 'a' in self.ground_truth:
            a_hat_seq = self.plant.a_seq[:, :self.t+1, :]
        else:
            y_seq = self.y[:, :self.t+1, :].clone() #[b,t,y]: y_{0}  ... y_{t}
            a_hat_seq = self.dfine.encoder(y_seq) #[b,t,a]: a_{0} ... a_{t}
        a_hat_t = a_hat_seq[:,-1,:]

        # Estimate dynamic latent from control inputs and history of manifold latents by Kalman filter
        if 'x' in self.ground_truth:
            x_hat_seq = self.plant.x_seq[:, :self.t+1, :]
        else:
            #TODO: Runs the KF from 0 to t every time which defeats the purpose. Do this recursively, try control.dlqe()
            u_seq = self.u[:, :self.t, :] #[b,t-1,u]: u_{0} ... u_{t-1}
            x_hat_seq = self.dfine.ldm(a=a_hat_seq, u=u_seq)[1] #[b,t,x]: x_{0} ... x_{t}
        x_hat_t = x_hat_seq[:,-1,:]

        # Log inferred latents
        self.a_hat[:, self.t, :] = a_hat_t
        self.x_hat[:, self.t, :] = x_hat_t
        self.a_hat_fwd[:, self.t, :], self.y_hat[:, self.t, :] = self.estimate_observation(x_hat_t)

        return x_hat_t


    def estimate_observation(self, x):
        """Generate the observation to which the estimated latents correspond, via x->a->y"""
        C = self.dfine.ldm.C #[a,x]
        a = (C @ x.unsqueeze(-1)).squeeze(-1) #[a,x]@[b,x,1]->[b,a]
        y = self.dfine.decoder(a) #[b,a]->[b,y]
        return a, y


    def plot_all(self, t_on=None, t_off=None, seq_num=0, max_plot_rows=7,
                 plot_vars = ['x', 'a', 'y', 'u'],
                 x_target=None, a_target=None, u_target=None, x=None, a=None): #only when plant is NonlinearStateSpaceModel

        var_names = {'x':'Dynamic latents',
                     'a':'Manifold latents',
                     'y':'Observations',
                     'u':'Control inputs'}

        max_dim = max([var.shape[-1] for var in (self.x_hat, self.a_hat, self.y_hat, self.u)])
        nrows = min(max_plot_rows, max_dim)
        ncols = len(plot_vars)
        fig, axs = plt.subplots(nrows, ncols, sharex=True, sharey=False, figsize=(3*ncols+1, 1.6*nrows+1))

        def plot(seq, target, varname, color, label):
            dim_v = seq.shape[-1]
            target = torch.full_like(seq, float('nan')) if target is None else target

            ax_num = plot_vars.index(varname)
            ax = np.expand_dims(axs[:dim_v, ax_num], axis=1)
            if dim_v < nrows:
                axs[dim_v-1, ax_num]._label_outer_xaxis(skip_non_rectangular_axes=False)
            for a in axs[dim_v:, ax_num]:
                a.axis('off')

            seq_i = seq[seq_num] if seq is not None else None
            target_i = target[seq_num] if target is not None else None

            plot_vs_time(seq_i, target_i, varname=varname, t_on=t_on, t_off=t_off,
                         ax=ax, max_N=nrows, legend=True, color=color, label=label)

            axs[0,ax_num].set_title(f'{var_names[varname]} ($n_{varname}$={dim_v})')
            return ax


        # Dynamic latents x
        if 'x' in plot_vars:
            plot(self.x_hat, self.x_hat_target, 'x', color='tab:blue', label=r'$\widehat{x}$')
            if x is not None or x_target is not None:
                x = torch.full_like(self.x_hat, float('nan')) if x is None else x
                plot(x, x_target, 'x', color='tab:orange', label='$x$')

        # Manifold latents a
        if 'a' in plot_vars:
            plot(self.a_hat,     self.a_hat_target,     'a', color='tab:blue', label=r'$\widehat{a}$')
            plot(self.a_hat_fwd, self.a_hat_fwd_target, 'a', color='tab:green', label=r'$\widetilde{a}$')
            if a is not None or a_target is not None:
                a = torch.full_like(self.a_hat, float('nan')) if a is None else a
                plot(a, a_target, 'a', color='tab:orange', label='$a$')

        # Observations y
        if 'y' in plot_vars:
            plot(self.y_hat, self.y_hat_target, 'y', color='tab:green', label=r'$\widehat{y}$')
            plot(self.y,     self.y_target,     'y', color='tab:orange', label='$y$')
            # plot(self.y*float('nan'),     self.y_target,     'y', color='tab:orange', label='$y$')


        # Control inputs u
        if 'u' in plot_vars:
            plot(self.u,     self.u_hat_target, 'u', color='tab:blue', label='$u$')
            if u_target is not None:
                _u = torch.full_like(self.u, float('nan'))
                plot(_u, u_target, 'u', color='tab:orange', label=None)

        fig.tight_layout()
        return fig, axs


###########
# Helpers #
###########
def make_closed_loop(plant, dfine, controller, copy_dfine=True, suppress_plant_noise=False, ground_truth=''):
    plant = deepcopy(plant)
    try:
        plant = plant.requires_grad_(False)
    except:
        pass

    if suppress_plant_noise:
        if plant.__class__.__name__ == 'NonlinearStateSpaceModel': #TEMP: allows changing class def
        # if isinstance(plant, NonlinearStateSpaceModel): #TODO: use this instead for robustness
            plant.Q_distr = plant.R_distr = plant.S_distr = None
        else:
            raise ValueError(f'Cannot turn off plant noise for {type(plant)}')

    if copy_dfine:
        dfine = deepcopy(dfine).requires_grad_(False)
    model = DFINE_Wrapper(dfine, plant, ground_truth=ground_truth)

    closed_loop = ClosedLoopSystem(plant, model, controller)
    return closed_loop
