import math

import matplotlib.pyplot as plt
import torch
import control

from DFINE import DFINE
from data.SSM import NonlinearStateSpaceModel


class LQGController():
    def __init__(self, plant, model):
        '''
        plant is an instance of DFINE or NonlinearStateSpaceModel
        model is an instance of DFINE
        '''
        self.model = model
        self.plant = plant


    def generate_observation(self, x=None, a=None):
        assert (x is None) != (a is None), 'Must specify exactly one of `x` or `a` for target'

        if a is None:
            a = self.plant.compute_manifold_latent(x.unsqueeze(-1)).squeeze(-1)

        if x is None: #TODO: not tested
            x = torch.linalg.solve(self.plant.C, a)
            return self.generate_observation(x=x)

        if isinstance(self.plant, DFINE):
            y = self.plant.decoder(a)
        elif isinstance(self.plant, NonlinearStateSpaceModel):
            y = self.plant.compute_observation(a, noise=False)
        else:
            raise ValueError('Invalid plant')

        return x, a, y


    def estimate_latents(self, y):
        """Get start and target points in model's latent space"""
        a_hat = self.model.encoder(y.unsqueeze(0)).squeeze()
        x_hat = torch.linalg.solve(self.model.ldm.C, a_hat) #x = Cinv @ a #TODO: is C generally invertible?
        return x_hat, a_hat


    def compute_lqr_gain(self, Q_scale=1, R_scale=1):
        """
        Compute infinite-horizon LQR feedback gain matrix G
        Cost function: J = \sum_t x^T Q x + u^T R u
        """
        A = self.model.ldm.A
        B = self.model.ldm.B
        C = self.model.ldm.C

        #effectively puts the penalty on a(t) with Q=I because a(t)=C*x(t)
        #need to do sqrt trick to ensure Q_lqr is symmetric due to numerical error
        Q_lqr = (math.sqrt(Q_scale)*C.T) @ (math.sqrt(Q_scale)*C)
        if (Q_lqr != Q_lqr.T).any():
            pass
        R_lqr = R_scale * torch.eye(self.model.dim_u)

        _, _, G = control.dare(A, B, Q_lqr, R_lqr) #TODO: should be computed for entire sequence A_t, B_t
        G = torch.tensor(G, dtype=torch.float32)
        return G


    def compute_steady_state_control(self, x_hat_ss):
        """Compute steady-state control input"""
        A = self.model.ldm.A
        B = self.model.ldm.B
        I = torch.eye(self.model.dim_x)
        u_ss = torch.linalg.pinv(B) @ (I-A) @ x_hat_ss
        return u_ss


    @torch.no_grad()
    def run_control(self, y_ss, x0=None, a0=None, Q=1, R=1, num_steps=100, t_ctrl_start=None, t_ctrl_stop=None):
        if t_ctrl_start is None:
            t_ctrl_start = -1
        if t_ctrl_stop is None:
            t_ctrl_stop = float('inf')

        # Compute control constants
        G = self.compute_lqr_gain(Q, R)
        x_hat_ss, a_hat_ss = self.estimate_latents(y_ss) #TODO: how to guarantee that this point in latent space actually corresponds to the desired target y_ss?
        u_ss = self.compute_steady_state_control(x_hat_ss)

        # Logging
        num_seqs = 1
        x     = torch.full((num_seqs, num_steps, self.plant.dim_x), torch.nan) #plant dynamic latent
        a     = torch.full((num_seqs, num_steps, self.plant.dim_a), torch.nan) #plant manifold latent (not used, just tracking for posterity)
        y     = torch.full((num_seqs, num_steps, self.plant.dim_y), torch.nan) #observations
        x_hat = torch.full((num_seqs, num_steps, self.model.dim_x), torch.nan) #Kalman filter estimate of dynamic latent (except at t=0, which is x_hat[0]=C_inv*f_enc(y))
        a_hat = torch.full((num_seqs, num_steps, self.model.dim_a), torch.nan) #model manifold latent
        u     = torch.full((num_seqs, num_steps, self.model.dim_u), torch.nan) #control input

        # Initial true and estimated states
        x[:,0,:], a[:,0,:], y[:,0,:] = self.generate_observation(x=x0, a=a0) #x0 xor a0 must be None
        x_hat[:,0,:], a_hat[:,0,:]   = self.estimate_latents(y[:,0,:])

        # Run controller
        for t in range(num_steps-1):
            # Compute LQR control input
            if t_ctrl_start <= t <= t_ctrl_stop:
                u[:,t,:] = -G @ (x_hat[:,t,:].squeeze() - x_hat_ss) + u_ss
            else:
                u[:,t,:] = 0

            # Step dynamics of plant with control input
            x[:,t+1,:], a[:,t+1,:], y[:,t+1,:] = self.plant.step(x[:,t,:].squeeze(), u[:,t,:].squeeze())

            # Estimate model manifold latent state from observation
            a_hat[:,t+1,:] = self.model.encoder(y[:,t+1,:])

            # Estimate dynamic latent from history of manifold latents by Kalman filter
            #TODO: do this recursively, try control.dlqe()
            x_hat[:,t+1,:] = self.model.ldm(a=a_hat[:, 1:t+2, :], u=u[:, :t+1, :])[1][:,-1,:]

        return {'x':x, 'a':a, 'y':y, 'u':u,
                'x_ss':None, 'a_ss':None, 'y_ss':y_ss, 'u_ss':u_ss,
                'x_hat':x_hat, 'a_hat':a_hat,
                'x_hat_ss':x_hat_ss, 'a_hat_ss':a_hat_ss,
                't_ctrl_start':t_ctrl_start, 't_ctrl_stop':t_ctrl_stop,
                'Q':Q, 'R':R}


    def plot_all(self, x, a, y, u,
                 x_ss, a_ss, y_ss, u_ss,
                 x_hat, a_hat,
                 x_hat_ss, a_hat_ss,
                 t_ctrl_start=-1, t_ctrl_stop=float('inf'),
                 Q=None, R=None):

        num_seqs, num_steps, dim_x = x.shape #num_seqs==1 for now
        dim_a = a_ss.shape[0]
        dim_y = y_ss.shape[0]
        dim_u = u_ss.shape[0]
        n_ax_rows = max(dim_x, dim_a, dim_y, dim_u)

        fig, ax = plt.subplots(n_ax_rows,4, sharex=True, figsize=(15,6))
        for i in range(n_ax_rows):
            if i < dim_x:
                ax[i,0].axhline(x_ss[i],      c='tab:blue',            label='$x_{ss}$ target (true)')
                ax[i,0].plot(   x[0,:,i],     c='tab:orange',          label='$x(t)$ plant')
                ax[i,0].axhline(x_hat_ss[i],  c='tab:blue',   ls='--', label='$\\widehat{x}_{ss}$ target (est)')
                ax[i,0].plot(   x_hat[0,:,i], c='tab:orange', ls='--', label='$\\widehat{x}(t)$ estimate (KF)')
                ax[i,0].set_ylabel(f'$x_{i+1}(t)$')
            # else:
            #     ax[i,0].axis('off')

            if i < dim_a:
                ax[i,1].axhline(a_ss[i],      c='tab:blue',            label='${a}_{ss}$ target (true)')
                ax[i,1].plot(   a[0,:,i],     c='tab:orange',          label='$a(t)$ plant')
                ax[i,1].axhline(a_hat_ss[i],  c='tab:blue',   ls='--', label='$\\widehat{a}_{ss}$ target (est)')
                ax[i,1].plot(   a_hat[0,:,i], c='tab:orange', ls='--', label='$\\widehat{a}(t)$ estimate ($f_{enc})$')
                ax[i,1].set_ylabel(f'$a_{i+1}(t)$')
            # else:
            #     ax[i,1].axis('off')

            if i < dim_u:
                ax[i,2].axhline(u_ss[i],  c='tab:blue',   label='$u_{ss}$ steady state')
                ax[i,2].plot(u[0,:,i],    c='tab:orange', label='$u(t)$ control input')
                ax[i,2].set_ylabel(f'$u_{i+1}((t)$')
            # else:
            #     ax[i,2].axis('off')

            if i < dim_y:
                ax[i,3].axhline(y_ss[i],  c='tab:blue',   label='$y_{ss}$ target')
                ax[i,3].plot(   y[0,:,i], c='tab:orange', label='$y(t)$ plant')
                ax[i,3].set_ylabel(f'$y_{i+1}(t)$')
            # else:
            #     ax[i,3].axis('off')

        for j,d in enumerate([dim_x, dim_a, dim_u, dim_y]):
            ax[0,j].legend()
            ax[-1,j].set_xlabel('Time')

        ax[0,0].set_title('Dynamic latents')
        ax[0,1].set_title('Manifold latents')
        ax[0,2].set_title('Control inputs')
        ax[0,3].set_title('Observations')

        for axis in ax.flatten():
            if axis not in ax[-1, :3]:
                if t_ctrl_start > 0:
                    axis.axvline(t_ctrl_start, color='k', ls='--')
                if t_ctrl_stop < num_steps:
                    axis.axvline(t_ctrl_stop,  color='k', ls='--')

        if Q is not None and R is not None:
            Q_str = '' if Q==1 else f'{Q}\\times '
            R_str = '' if R==1 else f'{R}\\times '
            fig.suptitle(f"$Q={Q_str}C^TC; R={R_str}I$")

        fig.tight_layout()
        return fig, ax
