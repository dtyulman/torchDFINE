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


###############
# Controllers #
###############
class Controller():
    """Base class all controllers must implement"""
    def __init__(self, dim_u, dim_x):
        pass

    def __call__(self, x):
        self.t += 1
        return 0


    def init_target(self, x_target):
        self.t = 0
        self.x_target = x_target
        num_seqs = x_target.shape[0]
        return torch.ones(num_seqs,self.dim_u)*float('nan')



class LinearQuadratic(Controller):
    """Base class for LQR"""
    def __init__(self, A, B, C, b=0, Q=1, R=1, Qf=None, u_min=-float('inf'), u_max=float('inf'), include_u_ss=True, penalize_obs=True):
        self.A = A
        self.B = B
        self.C = C
        self.b = b
        self.dim_x, self.dim_u = B.shape

        self.Q, self.R, self.Qf = self.make_cost_matrices(Q, R, Qf, penalize_obs)

        self.u_min = u_min
        self.u_max = u_max
        self.include_u_ss = include_u_ss


    def init_target(self, x_target):
        """
        x_target: [b,x]
        """
        super().init_target(x_target)
        self.u_target = self.compute_steady_state_control(x_target)
        return self.u_target


    def find_valid_equilibrium(self, x_target, alg=1):
        """Find the nearest equilibrium point to the desired target"""
        return find_nearest_solution(torch.eye(self.dim_x) - self.A, self.B, x_target, alg=alg)


    def compute_steady_state_control(self, x_ss):
        if not self.include_u_ss:
            num_seqs, dim_x = verify_shape(x_ss, [None, self.dim_x])
            return torch.zeros(num_seqs, self.dim_u)

        I = torch.eye(self.dim_x) #[x,x]
        IAx = ((I-self.A) @ x_ss.unsqueeze(-1) - self.b)

        pinvB = torch.linalg.pinv(self.B)
        u_ss = (pinvB @ IAx).squeeze(-1) #[u,x] @ ([x,x]@[b,x,1] - [b,x,1]) -> [b,u,1] -> [b,u]

        self.verify_steady_state_control(u_ss, IAx=IAx)

        if not ((self.u_min <= u_ss).all() and (u_ss <= self.u_max).all()):
            print(f'Warning: Steady state control input (min={u_ss.min()}, max={u_ss.max()}) '
                  f'outside of bounds (min={self.u_min}, max={self.u_max})')

        return u_ss


    def verify_steady_state_control(self, u_ss, x_ss=None, IAx=None):
        """
        u_ss: [b,u]
        x_ss: [b,x]
        """
        if IAx is None:
            I = torch.eye(self.dim_x) #[x,x]
            IAx = ((I-self.A) @ x_ss.unsqueeze(-1) - self.b)
        else:
            assert x_ss is None, 'Provide either x_ss or IAx but not both'

        Bu = self.B @ u_ss.unsqueeze(-1)
        if not torch.allclose(Bu, IAx):
            print(f'Warning: (I-A)x_ss not in range(B).\n  ||(I-A)x_ss - Bu_ss||={torch.norm(IAx-Bu, dim=1).squeeze().numpy()}')


    def make_cost_matrices(self, Q=1, R=1, Qf=None, penalize_obs=True):
        # Make Q matrix, penalizing state error
        if isinstance(Q, (float, int)):
            if penalize_obs:  #effectively puts the penalty on y(t) with Q=I because y = C @ x
                Q = Q * (self.C.T @ self.C) #[x,a]@[a,x]->[x,x]
            else:
                Q = Q * torch.eye(self.dim_x) #[x,x]

        # Make Qf matrix, penalizing state error at final timepoint
        if isinstance(Qf, (float, int)):
            if penalize_obs:
                Qf = Qf * (self.C.T @ self.C) #[x,a]@[a,x]->[x,x]
            else:
                Qf = Qf * torch.eye(self.dim_x) #[x,x]

        # Make R matrix, penalizing control input
        if isinstance(R, (float, int)):
            R = R * torch.eye(self.dim_u) #[u,u]

        return Q, R, Qf



class ConstantInputController(LinearQuadratic):
    def __init__(self, A, B, C, const=None):
        super().__init__(A, B, C)
        self.const = const * torch.ones(self.dim_u) if isinstance(const, (float, int)) else const


    def init_target(self, x_target):
        """
        x_target: [b,x]
        """
        super().init_target(x_target)
        if self.const is not None:
            print('Checking constant input...')
            self.verify_steady_state_control(self.const, self.x_target)
        return self.u_target


    def __call__(self, x):
        self.t += 1

        if self.const is not None:
            return self.const
        return self.u_target



class LinearQuadraticRegulator(LinearQuadratic):
    def __init__(self, *args, horizon=float('inf'), **kwargs):
        r"""
        System:
            x_{t+1} = A x_t + B u_t
            y_t = C x_t

        Control objective:
            if T < inf:
                J = x_T Qf x_T + \sum_{t=1}^{T-1} x_t Q x_t + u_t R u_t
            else:
                J = \sum_t x_t Q x_t + u_t R u_t

        u_min, u_max: ad-hoc hard bounds on the control input
        """
        super().__init__( *args, **kwargs)
        self.K = self.compute_lqr_gain(horizon)


    def compute_lqr_gain(self, horizon=float('inf')):
        """Compute LQR feedback gain matrix K"""
        if horizon < float('inf'): #finite-horizon LQR, time-varying K
            P = [None] * (horizon + 1)
            P[horizon] = self.Qf
            K = [None] * horizon
            for k in range(horizon-1, -1, -1):
                # K[k] = torch.linalg.inv(self.R + self.B.T @ P[k+1] @ self.B) @ (self.B.T @ P[k+1] @ self.A)
                K[k] = torch.linalg.solve(self.R + self.B.T @ P[k+1] @ self.B,
                                          self.B.T @ P[k+1] @ self.A) #same as above but more numerically robust
                P[k] = self.Q + self.A.T @ P[k+1] @ self.A - self.A.T @ P[k+1] @ self.B @ K[k]
                if K[k].isnan().any():
                    raise RuntimeError(f'K[{k}] contains NaNs')
            # print(f'norm(K[k])={[torch.linalg.matrix_norm(K[k], ord=2).item() for k in range(horizon-1, -1, -1)]}')

        else: #infinite-horizon LQR, constant K
            _, _, K = control.dare(self.A, self.B, self.Q, self.R) #[x,x],[x,u],[x,x],[u,u]->[u,x]
            K = torch.tensor(K, dtype=torch.get_default_dtype())
            # print(f'norm(K)={torch.linalg.matrix_norm(K, ord=2)}')

        return K


    def find_valid_equilibrium(self, x_target, closed_loop=False, alg=1):
        if closed_loop:
            assert not isinstance(self.K, list), 'Can only do closed-loop for constant K (infinite-horizon LQR)'
            M = torch.eye(self.dim_x) - (self.A - self.B@self.K)
            N = convert_to_tensor(sp.linalg.null_space(M))
            proj = N @ torch.linalg.pinv(N)
            return (proj @ x_target.unsqueeze(-1)).squeeze(-1)
        else:
            return find_nearest_solution(torch.eye(self.dim_x)-self.A, self.B, x_target, alg=alg)


    def plot_closed_loop_eigvals(self, verbose=True, return_eigvals=False):
        return plot_eigvals(self.A - self.B @ self.K,
                            title='eig($A-BK$)',
                            verbose=verbose,
                            return_eigvals=return_eigvals)


    def __call__(self, x):
        if isinstance(self.K, list): #finite-horizon
            K = self.K[self.t]
        else: #infinite-horizon
            K = self.K

        # print(f'norm(K[{self.t}])={torch.linalg.matrix_norm(K, ord=2)}')

        u = (-K @ (x - self.x_target).unsqueeze(-1)).squeeze(-1) + self.u_target #[u,x] @ [b,x,1] + [b,u] -> [b,u]
        u = torch.clip(u, min=self.u_min, max=self.u_max) #optional hard bounds

        self.t += 1

        return u



class MinimumEnergyController(LinearQuadratic):
    def __init__(self, A, B, b=0, num_steps=None, MPC_horizon=None, include_u_ss=True):
        self.A = A
        self.B = B
        self.b = b
        self.dim_x, self.dim_u = B.shape

        self.num_steps = num_steps
        self.horizon = MPC_horizon

        self.include_u_ss = include_u_ss
        self.u_min = -float('inf') #hack to make compatible with compute_steady_state_control
        self.u_max = float('inf')



    # def init_target(self, x_target):
    #     #TODO: can I somehow call super().super().init_target() without making it fragile to the number of super's?
    #     self.t = 0
    #     self.x_target = x_target
    #     num_seqs = x_target.shape[0]
    #     return torch.ones(num_seqs,self.dim_u)*float('nan')


    def compute_control_sequence(self, x0):
        """
        Minimize J = \\sum_{t=1}^T u(t)^T u(t) s.t. x(T)=x_target
        (equivalent to finite-horizon LQR with infinite terminal cost and R=I)

        x0: [b,x]
        u_seq: (N*m x 1)
        """

        horizon = min(self.horizon, self.num_steps-self.t) if self.horizon is not None else self.num_steps
        # horizon = self.horizon or self.num_steps


        # Compute A^N * x0
        A_N_x0 = torch.matrix_power(self.A, horizon) @ x0.unsqueeze(-1) #[x,x]@[b,x,1]->[b,x,1]

        # Compute target difference
        target_diff = self.x_target.unsqueeze(-1) - A_N_x0 #[b,x,1]-[b,x,1]

        # Compute optimal control sequence
        M_ctrb = torch.hstack([torch.matrix_power(self.A, i) @ self.B for i in range(horizon-1, -1, -1)]) #[x, u*horizon]
        u_seq = torch.linalg.pinv(M_ctrb) @ target_diff #[u*horizon,x]@[b,x,1]->[b,u*horizon,1]
        u_seq = u_seq.view(-1, horizon, self.dim_u)
        return u_seq


    def __call__(self, x):
        if self.horizon is None:
            if self.t == 0:
                self.u_seq = self.compute_control_sequence(x)
            u = self.u_seq[:,self.t,:]
        else:
            self.u_seq = self.compute_control_sequence(x)
            u = self.u_seq[:,0,:]

        self.t += 1
        return u + self.u_target



class Constrained_LQR_MPC(LinearQuadratic):
    """ Model Predictive Control with input constraints using Quadratic Programming."""
    #TODO: alt? https://www.do-mpc.com/en/latest/getting_started.html
    def __init__(self, A, B, C, Q, R, horizon, u_min, u_max):
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
        super().__init__(A,B,C, Q=Q,R=R, u_min=u_min, u_max=u_max)

        A = convert_to_numpy(self.A) #TODO: rewrite in pytorch
        B = convert_to_numpy(self.B)
        self.horizon = horizon

        # Build prediction matrices for the QP
        self.Q_ = np.kron(np.eye(horizon), self.Q)       # Block diagonal matrix of Q over the horizon
        self.R_ = np.kron(np.eye(horizon), self.R)       # Block diagonal matrix of R over the horizon

        # Build the augmented B matrix for the horizon
        self.B_ = np.zeros((self.dim_x * horizon, self.dim_u * horizon))
        for i in range(horizon):
            for j in range(i + 1):
                self.B_[i * self.dim_x:(i + 1) * self.dim_x, j * self.dim_u:(j + 1) * self.dim_u] = np.linalg.matrix_power(A, i - j) @ B

        # Build the augmented A matrix for initial state influence over horizon
        self.A_ = np.zeros((self.dim_x * horizon, self.dim_x))
        for i in range(horizon):
            self.A_[i * self.dim_x:(i + 1) * self.dim_x, :] = np.linalg.matrix_power(A, i + 1)

        # Define the QP matrices for the cost function
        self.P = matrix(2 * self.B_.T @ self.Q_ @ self.B_ + self.R_)

        # Inequality constraints for control bounds across the horizon
        self.G = matrix(np.vstack([np.eye(self.dim_u * horizon), -np.eye(self.dim_u * horizon)]))
        self.h = matrix(np.hstack([u_max * np.ones(self.dim_u * horizon), -u_min * np.ones(self.dim_u * horizon)]))


    def __call__(self, x):
        """
        x (np.array): Initial state vector.

        """
        num_seqs, dim_x = x.shape
        x = convert_to_numpy(x)

        u = np.full((num_seqs, self.dim_u), np.nan, dtype=x.dtype)
        for b in range(num_seqs):
            # Define the linear term in the QP objective
            x_target_b = convert_to_numpy(self.x_target[b,:])
            q = matrix(2 * self.B_.T @ self.Q_ @ self.A_ @ (x[b,:] - x_target_b))

            # Solve the QP to get the optimal control sequence
            sol = solvers.qp(self.P, q, self.G, self.h)
            u_sequence = np.array(sol['x']).reshape(self.horizon, self.dim_u)

            # Apply only the first control input in the sequence (receding horizon)
            u[b,:] = u_sequence[0]

        self.t += 1
        return convert_to_tensor(u)


###########
# Helpers #
###########
def find_nearest_solution(M, B, z, alg=1):
    """Given matrices M, B, find the x closest to z such that there exists a y for which Mx = By"""

    if alg == 1:
        x = torch.linalg.pinv(M) @ B @ torch.linalg.pinv(B) @ M @ z.unsqueeze(-1)

    elif alg == 2:
        # Compute u that minimizes the distance
        pinvM = torch.linalg.pinv(M)
        P_M = M @ pinvM  # projection onto row space of M
        u = torch.linalg.pinv(B) @ P_M @ z.unsqueeze(-1)

        # Compute z in the null space of M
        Q_M = torch.eye(P_M.shape[0]) - P_M  # projection onto null space of M
        x_null = Q_M @ z.unsqueeze(-1)

        # Final solution
        x = pinvM @ B @ u + x_null
    return x.squeeze(-1)


def make_controller(dfine, mode='LQR', clone_mats=True, **kwargs):
    A = dfine.ldm.A
    B = dfine.ldm.B
    C = dfine.ldm.C
    if clone_mats:
        A = A.detach().clone()
        B = B.detach().clone()
        C = C.detach().clone()

    if mode == 'LQR':
        controller = LinearQuadraticRegulator(A,B,C, **kwargs)
    elif mode == 'MPC':
        controller = Constrained_LQR_MPC(A,B,C, **kwargs)
    elif mode == 'const':
        controller = ConstantInputController(A,B,C, **kwargs)
    elif mode == 'MinE':
        controller = MinimumEnergyController(A,B, **kwargs)
    return controller
