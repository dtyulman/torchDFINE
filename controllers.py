import math
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import torch
import control
from cvxopt import matrix, solvers
solvers.options['show_progress'] = False  # Suppress solver output

from DFINE import DFINE
from data.SSM import NonlinearStateSpaceModel
from data.RNN import RNN, ReachRNN
from python_utils import verify_shape, convert_to_tensor, convert_to_numpy
from plot_utils import plot_vs_time



def make_closed_loop(plant, dfine, controller, suppress_plant_noise=False, ground_truth=''):
    plant = deepcopy(plant)
    try:
        plant = plant.requires_grad_(False)
    except:
        pass

    if suppress_plant_noise:
        if isinstance(plant, NonlinearStateSpaceModel):
            plant.Q_distr = plant.R_distr = plant.S_distr = None
        else:
            raise ValueError(f'Cannot turn off plant noise for {type(plant)}')
    dfine = deepcopy(dfine).requires_grad_(False)
    model = DFINE_Wrapper(dfine, plant, ground_truth=ground_truth)

    closed_loop = ClosedLoopSystem(plant, model, controller)
    return closed_loop


def make_controller(dfine, **kwargs):
    A = dfine.ldm.A.detach().clone()
    B = dfine.ldm.B.detach().clone()
    C = dfine.ldm.C.detach().clone()
    controller = LinearQuadraticRegulator(A,B,C, **kwargs)
    return controller



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

    class Controller():
        def __init__(self, dim_u, dim_x):
            pass
        def __call__(self, x):
            return u
        def set_target(self, x_target):
            self.x_target = x_target
    """
    def __init__(self, plant, model, controller):
        self.model = model
        self.plant = plant
        self.controller = controller


    def run(self, num_steps, y_target=None, x_hat_target=None, t_on=0, t_off=float('inf'), plant_init={}, aux_inputs=[{}]):

        # Defaults and convenience vars
        num_seqs = y_target.shape[0]

        if len(aux_inputs) == 1:
            aux_inputs *= num_steps
        else:
            assert len(aux_inputs) == num_steps
        #sanity check
        for aux_input_t in aux_inputs:
            for val in aux_input_t.values():
                assert val.shape[0] == num_seqs


        # Initialization and logging
        #plant
        self.plant.init_state(**plant_init, num_seqs=num_seqs, num_steps=num_steps)
        y0 = self.plant.get_observation()

        #model
        x_hat = self.model.init_state(y0, num_steps)
        x_hat_target = self.model.estimate_target(y_target, x_hat_target)

        #controller
        self.model.u_hat_target = self.controller.init_target(x_hat_target)

        # Run controller
        for t in range(0, num_steps-1):
            # Sanity check
            assert t == self.plant.t == self.model.t == self.controller.t+t_on

            # Compute control input
            u_t = 0
            if t_on <= t <= t_off:
                u_t = self.controller(x_hat) #[b,u]

            # Step dynamics of plant with control input
            full_state = self.plant.step(u=u_t, **aux_inputs[t]) #e.g. (x_{t+1}, a_{t+1}, y_{t+1}) for NL-SSM or (h_{t+1}, z_{t+1}) for RNN
            y = self.plant.get_observation() #y_{t+1} [b,y]

            # Estimate latents based on observation and input
            x_hat = self.model.estimate_state(y, u_t) #\hat{x}_{t+1} [b,x]




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
        self.a_hat_target = self.dfine.encoder(y_target.unsqueeze(1)) #[b,1,a]

        if x_hat_target is None:
            # Kalman update simplifies to C matrix inverse, assuming system has reached steady-state:
            #    x_ss = Ax_ss + Bu_ss + b + K(a_ss - C(Ax_ss + Bu_ss + b))
            # => x_ss = x_ss + K(a_ss - Cx_ss), since x_ss = Ax_ss + Bu_ss + b
            # => x_ss = Cinv @ a_ss
            C = self.dfine.ldm.C
            self.x_hat_target = (torch.linalg.pinv(C) @ self.a_hat_target.unsqueeze(-1)).squeeze(-1) #[x,a]@[b,1,a]->[b,1,x]
        else:
            assert len(x_hat_target.shape) == 2
            self.x_hat_target = x_hat_target.unsqueeze(1)

        # Estimate observation corresponding to latent target state estimate
        self.a_hat_fwd_target, self.y_hat_target = self.estimate_observation(self.x_hat_target)

        # Make sure shape is [b, dim_var]
        num_seqs = self.y_target.shape[0]
        for var in [self.y_target, self.x_hat_target, self.a_hat_fwd_target, self.y_hat_target]:
            if len(var.shape) == 3 and var.shape[1] == 1:
                var.squeeze_(1)
            elif len(var.shape) != 2 and var.shape[0] != num_seqs:
                raise RuntimeError('Something wrong with target dims')

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
            y_seq = self.y[:, :self.t+1, :] #[b,t,y]: y_{0}  ... y_{t}
            a_hat_seq = self.dfine.encoder(y_seq) #[b,t,a]: a_{0} ... a_{t}
            #TODO: DFINE.forward() reshapes it first, like this. Why?
            # a_hat = self.encoder(y.view(num_seq*num_steps, self.dim_y)).view(num_seq, num_steps, self.dim_a)

        # Estimate dynamic latent from control inputs and history of manifold latents by Kalman filter
        if 'x' in self.ground_truth:
            x_hat_seq = self.plant.x_seq[:, :self.t+1, :]
        else:
            #TODO: Runs the KF from 0 to t every time which defeats the purpose. Do this recursively, try control.dlqe()
            u_seq = self.u[:, :self.t, :] #[b,t-1,u]: u_{0} ... u_{t-1}
            # End-pad u_seq with dummy input u_{t} to compute prediction x_hat_{T+1|T} (which is never used)
            # u_{t}=nan so that it breaks if it's not dropped like it should be
            num_seqs, _, dim_u = u_seq.shape
            null = torch.full((num_seqs, 1, dim_u), torch.nan) #[b,1,u]
            u_seq = torch.cat([u_seq, null], dim=1) #[b,t,u]: u_{0}, ..., u_{t-1}, u_{t}
            x_hat_seq = self.dfine.ldm(a=a_hat_seq, u=u_seq)[1] #[b,t,x]: x_{0} ... x_{t}

        # Log inferred latents
        self.a_hat[:, self.t, :] = a_hat_seq[:,-1,:]
        self.x_hat[:, self.t, :] = x_hat_seq[:,-1,:]
        self.a_hat_fwd[:, self.t, :], self.y_hat[:, self.t, :] = self.estimate_observation(self.x_hat[:, self.t, :])

        return x_hat_seq[:,-1,:]


    def estimate_observation(self, x):
        """Generate the observation to which the estimated latents correspond, via x->a->y"""
        C = self.dfine.ldm.C #[a,x]
        a = (C @ x.unsqueeze(-1)).squeeze(-1) #[a,x]@[b,x,1]->[b,a]
        y = self.dfine.decoder(a) #[b,a]->[b,y]
        return a, y


    def plot_all(self, t_on=None, t_off=None, seq_num=0, max_plot_rows=7,
                 x_target=None, a_target=None, u_target=None, x=None, a=None): #only when plant is NonlinearStateSpaceModel

        var_names = ['x', 'a', 'y', 'u']
        var_names_long = ['Dynamic latents', 'Manifold latents', 'Observations', 'Control inputs']

        max_dim = max([var.shape[-1] for var in (self.x_hat, self.a_hat, self.y_hat, self.u)])
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
        plot(self.x_hat, self.x_hat_target, 'x', color='tab:blue', label=r'$\widehat{x}$')
        plot(self.a_hat, self.a_hat_target, 'a', color='tab:blue', label=r'$\widehat{a}$')
        plot(self.u,     self.u_hat_target, 'u', color='tab:blue', label='$u$')

        #decoded vars
        plot(self.y_hat,     self.y_hat_target,     'y', color='tab:green', label=r'$\widehat{y}$')
        plot(self.a_hat_fwd, self.a_hat_fwd_target, 'a', color='tab:green', label=r'$\widetilde{a}$')

        #plant vars
        plot(self.y,         self.y_target,         'y', color='tab:orange', label='$y$')

        if u_target is not None:
            _u = torch.full_like(u, float('nan'))
            plot(_u, u_target, 'u', color='tab:orange', label=None)
        if a is not None or a_target is not None:
            a = torch.full_like(self.a_hat, float('nan')) if a is None else a
            a_target = torch.full_like(a_hat_target, float('nan')) if a_target is None else a_target
            plot(a, a_target, 'a', color='tab:orange', label='$a$')
        if x is not None or x_target is not None:
            x = torch.full_like(self.x_hat, float('nan')) if x is None else x
            x_target = torch.full_like(self.x_hat_target, float('nan')) if x_target is None else x_target
            plot(x, x_target, 'x', color='tab:orange', label='$x$')

        fig.tight_layout()
        return fig, axs



class LinearQuadraticRegulator():
    def __init__(self, A, B, C, b=0, Q=1, R=1, F=None, T=float('inf'), u_min=-float('inf'), u_max=float('inf')):
        r"""
        System:
            x_{t+1} = A x_t + b + B u_t
            y_t = C x_t

        Control objective:
            if T < inf:
                J = x_T F x_T + \sum_{t=1}^{T-1} x_t Q x_t + u_t R u_t
            else:
                J = \sum_t x_t Q x_t + u_t R u_t

        u_min, u_max: ad-hoc hard bounds on the control input
        """
        self.A = A
        self.b = b
        self.B = B
        self.C = C
        self.dim_x, self.dim_u = B.shape

        self.Q, self.R, self.F = self.make_cost_matrices(Q, R, F)
        self.u_min = u_min
        self.u_max = u_max

        self.K = self.compute_lqr_gain(T)


    def make_cost_matrices(self, Q=1, R=1, F=None, penalize_obs=True):
        # Make Q matrix, penalizing state error
        if isinstance(Q, (float, int)):
            if penalize_obs:  #effectively puts the penalty on y(t) with Q=I because y = C @ x
                Q = Q * (self.C.T @ self.C) #[x,a]@[a,x]->[x,x]
            else:
                Q = Q * torch.eye(self.dim_x) #[x,x]

        # Make F matrix, penalizing state error at final timepoint
        if isinstance(F, (float, int)):
            if penalize_obs:
                F = F * (self.C.T @ self.C) #[x,a]@[a,x]->[x,x]
            else:
                F = F * torch.eye(self.dim_x) #[x,x]

        # Make R matrix, penalizing control input
        if isinstance(R, (float, int)):
            R = R * torch.eye(self.dim_u) #[u,u]

        return Q, R, F


    def compute_lqr_gain(self, T=float('inf')):
        """Compute LQR feedback gain matrix K"""
        if T < float('inf'): #finite-horizon LQR, time-varying K
            P = [None] * (T + 1)
            P[T] = self.F
            K = [None] * T
            for k in range(T-1, -1, -1):
                K[k] = torch.linalg.inv(self.R + self.B.T @ P[k+1] @ self.B) @ (self.B.T @ P[k+1] @ self.A)
                P[k] = self.Q + self.A.T @ P[k+1] @ self.A - self.A.T @ P[k+1] @ self.B @ K[k]
        else: #infinite-horizon LQR, constant K
            _, _, K = control.dare(self.A, self.B, self.Q, self.R) #[x,x],[x,u],[x,x],[u,u]->[u,x]
            K = torch.tensor(K, dtype=torch.get_default_dtype())

        return K


    def compute_steady_state_control(self, x_ss):
        I = torch.eye(self.dim_x) #[x,x]
        Binv = torch.linalg.pinv(self.B)
        u_ss = Binv @ ((I-self.A) @ x_ss.unsqueeze(-1) - self.b) #[u,x] @ ([x,x]@[b,x,1] - [b,x,1]) -> [b,u,1]

        if not ((self.u_min <= u_ss).all() and (u_ss <= self.u_max).all()):
            print(f'Warning: Steady state control input (min={u_ss.min()}, max={u_ss.max()}) '
                  f'outside of bounds (min={self.u_min}, max={self.u_max})')

        return u_ss.squeeze(-1) #[b,u]


    def compute_cost(self, x, x_target, u, u_target=0):
        """
        x: [b,t,x]
        x_target: [b,t,x]
        u: [b,t-1,u]
        u_target: [b,t-1,u]
        """
        x_err = x - x_target
        u_err = u - u_target

        xT_Q_x = x_err.unsqueeze(-2) @ self.Q @ x_err.unsqueeze(-1) #[b,t,1,x]@[x,x]@[b,t,x,1]->[b,t,1,1]
        uT_R_u = u_err.unsqueeze(-2) @ self.R @ u_err.unsqueeze(-1) #[b,t,1,u]@[u,u]@[b,t,u,1]->[b,t,1,1]

        return xT_Q_x.mean() + uT_R_u.mean()


    def init_target(self, x_target):
        """
        x_target: [b,x]
        """
        self.x_target = x_target
        self.u_target = self.compute_steady_state_control(x_target)
        self.t = 0

        return self.u_target


    def __call__(self, x):
        if isinstance(self.K, list): #finite-horizon
            K = self.K[self.t]
        else: #infinite-horizon
            K = self.K

        u = (-K @ (x - self.x_target).unsqueeze(-1)).squeeze(-1) + self.u_target #[u,x] @ [b,x,1] + [b,u] -> [b,u]
        u = torch.clip(u, min=self.u_min, max=self.u_max) #optional hard bounds

        self.t += 1

        return u



class Constrained_LQR_MPC:
    """ Model Predictive Control with input constraints using Quadratic Programming."""
    def __init__(self, x_ss, A, B, Q, R, horizon, u_min, u_max):
        """
        Inputs:
            A (np.array): System dynamics matrix.
            B (np.array): Control input matrix.
            Q (np.array): State cost matrix.
            R (np.array): Control input cost matrix.
            horizon (int): Prediction horizon for the MPC.
            u_min (float): Minimum control input.
            u_max (float): Maximum control input.
        """
        #Target
        self.x_ss = convert_to_numpy(x_ss)


        # Dimensions
        self.n = n = A.shape[0]  # State dimension
        self.m = m = B.shape[1]  # Control input dimension
        self.horizon = horizon

        # Build prediction matrices for the QP
        self.Q = np.kron(np.eye(horizon), Q)       # Block diagonal matrix of Q over the horizon
        self.R = np.kron(np.eye(horizon), R)       # Block diagonal matrix of R over the horizon

        # Build the augmented B matrix for the horizon
        self.B = np.zeros((n * horizon, m * horizon))
        for i in range(horizon):
            for j in range(i + 1):
                self.B[i * n:(i + 1) * n, j * m:(j + 1) * m] = np.linalg.matrix_power(A, i - j) @ B

        # Build the augmented A matrix for initial state influence over horizon
        self.A = np.zeros((n * horizon, n))
        for i in range(horizon):
            self.A[i * n:(i + 1) * n, :] = np.linalg.matrix_power(A, i + 1)

        # Define the QP matrices for the cost function
        self.P = matrix(2 * self.B.T @ self.Q @ self.B + self.R)

        # Inequality constraints for control bounds across the horizon
        self.G = matrix(np.vstack([np.eye(m * horizon), -np.eye(m * horizon)]))
        self.h = matrix(np.hstack([u_max * np.ones(m * horizon), -u_min * np.ones(m * horizon)]))


    def __call__(self, x):
        """
        x (np.array): Initial state vector.

        """
        B,N = x.shape
        x = convert_to_numpy(x)

        u = np.empty((B, self.m))
        for b in range(B):
            # Define the linear term in the QP objective
            q = matrix(2 * self.B.T @ self.Q @ self.A @ (x[b,:] - self.x_ss[b,:]))

            # Solve the QP to get the optimal control sequence
            sol = solvers.qp(self.P, q, self.G, self.h)
            u_sequence = np.array(sol['x']).reshape(self.horizon, self.m)

            # Apply only the first control input in the sequence (receding horizon)
            u[b,:] = u_sequence[0]

        return convert_to_tensor(u)
