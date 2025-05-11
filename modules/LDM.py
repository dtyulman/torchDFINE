'''
Copyright (c) 2023 University of Southern California
See full notice in LICENSE.md
Hamidreza Abbaspourazad*, Eray Erturk* and Maryam M. Shanechi
Shanechi Lab, University of Southern California
'''

import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal


class LDM(nn.Module):
    '''
    Linear Dynamical Model backbone for DFINE. This module is used for smoothing and filtering
    given a batch of trials/segments/time-series.

    LDM equations are as follows:
    x_{t+1} = A*x_t + B*u_t + w_t , cov(w_t) = W
    a_t     = C*x_t + r_t         , cov(r_t) = R
    '''

    def __init__(self, **kwargs):
        '''
        Initializer for an LDM object. Note that LDM is a subclass of torch.nn.Module.

        Parameters
        ------------
        - dim_x: int, Dimensionality of dynamic latent factors, default None
        - dim_u: int, Dimensionality of control input, default None
        - dim_a: int, Dimensionality of manifold latent factors, default None
        - is_W_trainable: bool, Whether dynamics noise covariance matrix (W) is learnt or not, default True
        - is_R_trainable: bool, Whether observation noise covariance matrix (R) is learnt or not, default True
        - A: torch.Tensor, shape: (self.dim_x, self.dim_x), State transition matrix of LDM, default identity
        - B: torch.Tensor, shape: (num_seq, num_steps, dim_x, dim_u) or (dim_x, dim_u), Control-input matrix of LDM, default identity
        - C: torch.Tensor, shape: (self.dim_a, self.dim_x), Observation matrix of LDM, default identity
        - D: torch.Tensor, shape: (self.dim_a, self.dim_u), Feed-through matrix of LDM, default identity (it is learned only if fit_D_matrix flag is True)
        - mu_0: torch.Tensor, shape: (self.dim_x, ), Dynamic latent factor estimate initial condition (x_{0|-1}) for Kalman filtering, default zeros
        - Lambda_0: torch.Tensor, shape: (self.dim_x, self.dim_x), Dynamic latent factor estimate error covariance initial condition (P_{0|-1}) for Kalman Filtering, default identity
        - W_log_diag: torch.Tensor, shape: (self.dim_x, ), Log-diagonal of process noise covariance matrix (W, therefore it is diagonal and PSD), default ones
        - R_log_diag: torch.Tensor, shape: (self.dim_a, ), Log-diagonal of observation noise covariance matrix  (R, therefore it is diagonal and PSD), default ones
        '''

        super(LDM, self).__init__()

        self.dim_x = kwargs.pop('dim_x', None)
        self.dim_u = kwargs.pop('dim_u', None)
        self.dim_a = kwargs.pop('dim_a', None)

        self.is_W_trainable = kwargs.pop('is_W_trainable', True)
        self.is_R_trainable = kwargs.pop('is_R_trainable', True)
        self.fit_D_matrix = kwargs.pop('fit_D_matrix', False)

        # Get initial values for LDM parameters
        self.A = kwargs.pop('A', torch.eye(self.dim_x, self.dim_x))
        self.B = kwargs.pop('B', torch.eye(self.dim_x, self.dim_u))
        self.C = kwargs.pop('C', torch.eye(self.dim_a, self.dim_x))
        self.D = kwargs.pop('D', torch.eye(self.dim_a, self.dim_u)) if self.fit_D_matrix else torch.zeros(self.dim_a, self.dim_u)

        # Get KF initial conditions
        self.mu_0 = kwargs.pop('mu_0', torch.zeros(self.dim_x))
        self.Lambda_0 = kwargs.pop('Lambda_0', torch.eye(self.dim_x, self.dim_x))

        # Get initial process and observation noise parameters
        self.W_log_diag = kwargs.pop('W_log_diag', torch.ones(self.dim_x))
        self.R_log_diag = kwargs.pop('R_log_diag', torch.ones(self.dim_a))

        # Register trainable parameters to module
        self._register_params()


    def __repr__(self):
        W, R = self._get_covariance_matrices()
        r = (f'A={self.A.detach().numpy()}\n'
             f'B={self.B.detach().numpy()}\n'
             f'W={W.detach().numpy()}\n'
             '--\n'

             f'C={self.C.detach().numpy()}\n'
             f'D={self.D.detach().numpy() if self.fit_D_matrix else None}\n'
             f'R={R.detach().numpy()}')
        return r


    def _register_params(self):
        '''
        Registers the learnable LDM parameters as nn.Parameters
        '''

        # Check if LDM matrix shapes are consistent
        self._check_matrix_shapes()

        # Register LDM parameters
        self.A = torch.nn.Parameter(self.A, requires_grad=True)
        self.B = torch.nn.Parameter(self.B, requires_grad=True)
        self.C = torch.nn.Parameter(self.C, requires_grad=True)
        self.D = torch.nn.Parameter(self.D, requires_grad=self.fit_D_matrix)

        self.W_log_diag = torch.nn.Parameter(self.W_log_diag, requires_grad=self.is_W_trainable)
        self.R_log_diag = torch.nn.Parameter(self.R_log_diag, requires_grad=self.is_R_trainable)

        self.mu_0 = torch.nn.Parameter(self.mu_0, requires_grad=True)
        self.Lambda_0 = torch.nn.Parameter(self.Lambda_0, requires_grad=True)


    def _check_matrix_shapes(self):
        '''
        Checks whether LDM parameters have the correct shapes, which are defined above in the constructor
        '''

        # Check model matrix shapes
        assert self.A.shape == (self.dim_x, self.dim_x), 'Shape of A matrix must be (dim_x, dim_x)!'
        assert self.B.shape == (self.dim_x, self.dim_u), 'Shape of B matrix must be (dim_x, dim_u)!'
        assert self.C.shape == (self.dim_a, self.dim_x), 'Shape of C matrix must be (dim_a, dim_x)!'
        assert self.D.shape == (self.dim_a, self.dim_u), 'Shape of D matrix must be (dim_a, dim_u)!'

        # Check mu_0 matrix's shape
        if len(self.mu_0.shape) != 1:
            self.mu_0 = self.mu_0.view(-1, )
        assert self.mu_0.shape == (self.dim_x, ), 'Shape of mu_0 matrix must be (dim_x, )!'

        # Check Lambda_0 matrix's shape
        assert self.Lambda_0.shape == (self.dim_x, self.dim_x), 'Shape of Lambda_0 matrix must be (dim_x, dim_x)!'

        # Check W_log_diag matrix's shape
        if len(self.W_log_diag.shape) != 1:
            self.W_log_diag = self.W_log_diag.view(-1, )
        assert self.W_log_diag.shape == (self.dim_x, ), 'Shape of W_log_diag matrix must be (dim_x, )!'

        # Check R_log_diag matrix's shape
        if len(self.R_log_diag.shape) != 1:
            self.R_log_diag = self.R_log_diag.view(-1, )
        assert self.R_log_diag.shape == (self.dim_a, ), 'Shape of R_log_diag matrix must be (dim_x, )!'


    def _get_covariance_matrices(self):
        '''
        Get the process and observation noise covariance matrices from log-diagonals.

        Returns:
        ------------
        - W: torch.Tensor, shape: (self.dim_x, self.dim_x), Process noise covariance matrix
        - R: torch.Tensor, shape: (self.dim_a, self.dim_a), Observation noise covariance matrix
        '''

        W = torch.diag(torch.exp(self.W_log_diag))
        R = torch.diag(torch.exp(self.R_log_diag))
        return W, R


    def compute_forwards(self, a, u, mask=None):
        '''
        Performs the forward iteration of causal flexible Kalman filtering, given a batch of trials/segments/time-series

        Parameters:
        ------------
        - a: torch.Tensor, shape: (num_seq, num_steps, dim_a),
            Batch of projected manifold latent factors (outputs of encoder; nonlinear manifold embedding step)
        - u: torch.Tensor, shape: (num_seq, num_steps, dim_u),
            Batch of control input vectors
        - mask: torch.Tensor, shape: (num_seq, num_steps, 1),
            Mask input which shows whether observations at each timestep exists (1) or are missing (0)

        Returns:
        ------------
        - mu_pred_all: torch.Tensor, shape: (num_steps, num_seq, dim_x),
            Dynamic latent factor predictions (t+1|t) where first index of the second dimension has x_{1|0}
        - mu_t_all: torch.Tensor, shape: (num_steps, num_seq, dim_x),
            Dynamic latent factor filtered estimates (t|t) where first index of the second dimension has x_{0|0}
        - Lambda_pred_all: torch.Tensor, shape: (num_steps, num_seq, dim_x, dim_x),
            Dynamic latent factor estimation error covariance predictions (t+1|t) where first index of the second dimension has P_{1|0}
        - Lambda_t_all: torch.Tensor, shape: (num_steps, num_seq, dim_x, dim_x),
            Dynamic latent factor estimation error covariance filtered estimates (t|t) where first index of the second dimension has P_{0|0}
        '''

        num_seq, num_steps, _ = a.shape #[b,t,a]

        if mask is None:
            mask = torch.ones(num_seq, num_steps)

        # Make sure that mask is 3D (last axis is 1-dimensional)
        if len(mask.shape) != len(a.shape):
            mask = mask.unsqueeze(dim=-1) # (num_seq, num_steps, 1)

        # To make sure we do not accidentally use the real outputs in the steps with missing values, set them to a dummy value, e.g., 0.
        # The dummy values of observations at masked points are irrelevant because:
        # Kalman disregards the observations by setting Kalman Gain to 0 in K = torch.mul(K, mask[:, t, ...].unsqueeze(dim=1)) @ line 205
        a_masked = torch.mul(a, mask) # (num_seq, num_steps, dim_a) x (num_seq, num_steps, 1)

        # Initialize mu_0 and Lambda_0
        mu_pred = self.mu_0.unsqueeze(dim=0).repeat(num_seq, 1) # (num_seq, dim_x)
        Lambda_pred = self.Lambda_0.unsqueeze(dim=0).repeat(num_seq, 1, 1) # (num_seq, dim_x, dim_x)

        # Filtered and predicted estimates/error covariance. NOTE: The last time-step of the prediction has T+1|T, which may not be of interest
        mu_t_all = [] #mu_t_all[t,i,:] = mu_{t|t} for t=0..T-1
        Lambda_t_all = []
        mu_pred_all = [] #mu_pred_all[t,i,:] = mu_{t+1|t} for t=0..T-1
        Lambda_pred_all = []

        # Get covariance matrices
        W, R = self._get_covariance_matrices()

        for t in range(num_steps):
            # Obtain residual
            a_pred = (self.C @ mu_pred.unsqueeze(dim=-1)).squeeze(dim=-1) # (num_seq, dim_a)
            r = a_masked[:, t, ...] - a_pred # (num_seq, dim_a)
            if self.fit_D_matrix:
                r = r - (self.D @ u[:,t,...].unsqueeze(dim=-1)).squeeze(dim=-1) # (num_seq, dim_a)

            # Project system uncertainty into measurement space, get Kalman Gain
            S = self.C @ Lambda_pred @ self.C.T + R # (num_seq, dim_a, dim_a)
            S_inv = torch.inverse(S) # (num_seq, dim_a, dim_a)
            K = Lambda_pred @ self.C.T @ S_inv # (num_seq, dim_x, dim_a)
            K = torch.mul(K, mask[:, t, ...].unsqueeze(dim=1))  # (num_seq, dim_x, dim_a) x (num_seq, 1,  1)

            # Get current mu and Lambda
            mu_t = mu_pred + (K @ r.unsqueeze(dim=-1)).squeeze(dim=-1) # (num_seq, dim_x)
            I_KC = torch.eye(self.dim_x) - K @ self.C # (num_seq, dim_x, dim_x)
            Lambda_t = I_KC @ Lambda_pred # (num_seq, dim_x, dim_x)

            # Prediction
            if t < num_steps-1:
                u_t = u[:, t, ...]
            else:
                #last timestep of predictions will be dropped since it's T+1|T so give it dummy input
                u_t = torch.full((num_seq, self.dim_u), torch.nan)

            mu_pred = (self.A @ mu_t.unsqueeze(dim=-1) + self.B @ u_t.clone().unsqueeze(dim=-1)).squeeze(dim=-1) #(num_seq, dim_x, dim_x) x (num_seq, dim_x, 1) + (num_seq, dim_x, dim_u) x (num_seq, dim_u, 1) --> (num_seq, dim_x, 1) --> (num_seq, dim_x)
            Lambda_pred = self.A @ Lambda_t @ self.A.T + W #(num_seq, dim_x, dim_x) x (num_seq, dim_x, dim_x) x (num_seq, dim_x, dim_x) --> (num_seq, dim_x, dim_x)

            # Store predictions and updates
            mu_pred_all.append(mu_pred)
            mu_t_all.append(mu_t)
            Lambda_pred_all.append(Lambda_pred)
            Lambda_t_all.append(Lambda_t)

        mu_pred_all = torch.stack(mu_pred_all, dim=0) #[t,b,x]
        mu_t_all = torch.stack(mu_t_all, dim=0) #[t,b,x]
        Lambda_pred_all = torch.stack(Lambda_pred_all, dim=0) #[t,b,x,x]
        Lambda_t_all = torch.stack(Lambda_t_all, dim=0) #[t,b,x,x]

        return mu_pred_all, mu_t_all, Lambda_pred_all, Lambda_t_all


    def filter(self, a, u=None, mask=None):
        '''
        Performs Kalman Filtering

        Parameters:
        ------------
        - a: torch.Tensor, shape: (num_seq, num_steps, dim_a),
            Batch of projected manifold latent factors (outputs of encoder; nonlinear manifold embedding step)
        - u: torch.Tensor, shape: (num_seq, num_steps, dim_u),
            Batch of control input vectors
        - mask: torch.Tensor, shape: (num_seq, num_steps, 1),
            Mask input which shows whether observations at each timestep exists (1) or are missing (0)

        Returns:
        ------------
        - mu_pred_all: torch.Tensor, shape: (num_seq, num_steps, dim_x),
            Dynamic latent factor predictions (t+1|t) where first index of the second dimension has x_{1|0}
        - mu_t_all: torch.Tensor, shape: (num_seq, num_steps, dim_x),
            Dynamic latent factor filtered estimates (t|t) where first index of the second dimension has x_{0|0}
        - Lambda_pred_all: torch.Tensor, shape: (num_seq, num_steps, dim_x, dim_x),
            Dynamic latent factor estimation error covariance predictions (t+1|t) where first index of the second dimension has P_{1|0}
        - Lambda_t_all: torch.Tensor, shape: (num_seq, num_steps, dim_x, dim_x),
            Dynamic latent factor estimation error covariance filtered estimates (t|t) where first index of the second dimension has P_{0|0}
        '''

        # Run the forward iteration
        mu_pred_all, mu_t_all, Lambda_pred_all, Lambda_t_all = self.compute_forwards(a=a, u=u, mask=mask)

        # Swap num_seq and num_steps dimensions
        mu_pred_all = torch.permute(mu_pred_all, (1, 0, 2))
        mu_t_all = torch.permute(mu_t_all, (1, 0, 2))
        Lambda_pred_all = torch.permute(Lambda_pred_all, (1, 0, 2, 3))
        Lambda_t_all = torch.permute(Lambda_t_all, (1, 0, 2, 3))

        return mu_pred_all, mu_t_all, Lambda_pred_all, Lambda_t_all


    def compute_backwards(self, mu_pred_all, mu_t_all, Lambda_pred_all, Lambda_t_all):
        '''
        Performs backward iteration for Rauch-Tung-Striebel (RTS) Smoother

        Parameters:
        ------------
        - mu_pred_all: torch.Tensor, shape: (num_seq, num_steps, dim_x),
            Dynamic latent factor predictions (t+1|t) where first index of the second dimension has x_{1|0}
        - mu_t_all: torch.Tensor, shape: (num_seq, num_steps, dim_x),
            Dynamic latent factor filtered estimates (t|t) where first index of the second dimension has x_{0|0}
        - Lambda_pred_all: torch.Tensor, shape: (num_seq, num_steps, dim_x, dim_x),
            Dynamic latent factor estimation error covariance predictions (t+1|t) where first index of the second dimension has P_{1|0}
        - Lambda_t_all: torch.Tensor, shape: (num_seq, num_steps, dim_x, dim_x),
            Dynamic latent factor estimation error covariance filtered estimates (t|t) where first index of the second dimension has P_{0|0}

        Returns:
        ------------
        - mu_back_all: torch.Tensor, shape: (num_steps, num_seq, dim_x),
            Dynamic latent factor smoothed estimates (t|T) where first index of the second dimension has x_{0|T}
        - Lambda_back_all: torch.Tensor, shape: (num_steps, num_seq, dim_x, dim_x),
            Dynamic latent factor estimation error covariance smoothed estimates (t|T) where first index of the second dimension has P_{0|T}
        '''

        # Get number of steps and number of trials
        num_steps, num_seq, _ = mu_pred_all.shape

        # Create empty arrays for smoothed dynamic latent factors and error covariances
        mu_back_all = torch.zeros((num_steps, num_seq, self.dim_x), device=mu_pred_all.device) # (num_steps, num_seq, dim_x)
        Lambda_back_all = torch.zeros((num_steps, num_seq, self.dim_x, self.dim_x), device=mu_pred_all.device) # (num_steps, num_seq, dim_x, dim_x)

        # Last smoothed estimation is equivalent to the filtered estimation
        mu_back_all[-1, ...] = mu_t_all[-1, ...]
        Lambda_back_all[-1, ...] = Lambda_t_all[-1, ...]

        # Initialize iterable parameter
        mu_back = mu_t_all[-1, ...]
        Lambda_back = Lambda_back_all[-1, ...]

        for t in range(num_steps-2, -1, -1): # iterate loop over reverse time: T-2, T-3, ..., 0, where the last time-step is T-1
            J_t = Lambda_t_all[t, ...] @ self.A.T @ torch.inverse(Lambda_pred_all[t, ...]) # (num_seq, dim_x, dim_x) x (num_seq, dim_x, dim_x) x (num_seq, dim_x, dim_x)
            mu_back = mu_t_all[t, ...] + (J_t @ (mu_back - mu_pred_all[t, ...]).unsqueeze(dim=-1)).squeeze(dim=-1) # (num_seq, dim_x) + (num_seq, dim_x, dim_x) x (num_seq, dim_x)

            Lambda_back = Lambda_t_all[t, ...] + J_t @ (Lambda_back - Lambda_pred_all[t, ...]) @ torch.permute(J_t, (0, 2, 1)) # (num_seq, dim_x, dim_x)

            mu_back_all[t, ...] = mu_back
            Lambda_back_all[t, ...] = Lambda_back

        return mu_back_all, Lambda_back_all


    def smooth(self, a, u=None, mask=None):
        '''
        Performs Rauch-Tung-Striebel (RTS) Smoothing

        Parameters:
        ------------
        - a: torch.Tensor, shape: (num_seq, num_steps, dim_a),
            Batch of projected manifold latent factors (outputs of encoder; nonlinear manifold embedding step)
        - u: torch.Tensor, shape: (num_seq, num_steps, dim_u),
            Batch of control input vectors
        - mask: torch.Tensor, shape: (num_seq, num_steps, 1),
            Mask input which shows whether observations at each timestep exists (1) or are missing (0)

        Returns:
        ------------
        - mu_pred_all: torch.Tensor, shape: (num_seq, num_steps, dim_x),
            Dynamic latent factor predictions (t+1|t) where first index of the second dimension has x_{1|0}
        - mu_t_all: torch.Tensor, shape: (num_seq, num_steps, dim_x),
            Dynamic latent factor filtered estimates (t|t) where first index of the second dimension has x_{0|0}
        - mu_back_all: torch.Tensor, shape: (num_seq, num_steps, dim_x),
            Dynamic latent factor smoothed estimates (t|T) where first index of the second dimension has x_{0|T}
        - Lambda_pred_all: torch.Tensor, shape: (num_seq, num_steps, dim_x, dim_x),
            Dynamic latent factor estimation error covariance predictions (t+1|t) where first index of the second dimension has P_{1|0}
        - Lambda_t_all: torch.Tensor, shape: (num_seq, num_steps, dim_x, dim_x),
            Dynamic latent factor estimation error covariance filtered estimates (t|t) where first index of the second dimension has P_{0|0}
        - Lambda_back_all: torch.Tensor, shape: (num_seq, num_steps, dim_x, dim_x),
            Dynamic latent factor estimation error covariance smoothed estimates (t|T) where first index of the second dimension has P_{0|T}
        '''

        mu_pred_all, mu_t_all, Lambda_pred_all, Lambda_t_all = self.compute_forwards(a=a, u=u, mask=mask)
        mu_back_all, Lambda_back_all = self.compute_backwards(mu_pred_all=mu_pred_all,
                                                              mu_t_all=mu_t_all,
                                                              Lambda_pred_all=Lambda_pred_all,
                                                              Lambda_t_all=Lambda_t_all)

        # Swap num_seq and num_steps dimensions
        mu_pred_all = torch.permute(mu_pred_all, (1, 0, 2))
        mu_t_all = torch.permute(mu_t_all, (1, 0, 2))
        mu_back_all = torch.permute(mu_back_all, (1, 0, 2))

        Lambda_pred_all = torch.permute(Lambda_pred_all, (1, 0, 2, 3))
        Lambda_t_all = torch.permute(Lambda_t_all, (1, 0, 2, 3))
        Lambda_back_all = torch.permute(Lambda_back_all, (1, 0, 2, 3))

        return mu_pred_all, mu_t_all, mu_back_all, Lambda_pred_all, Lambda_t_all, Lambda_back_all


    def forward(self, a, u=None, mask=None, do_smoothing=False):
        '''
        Forward pass function for LDM Module

        Parameters:
        ------------
        - a: torch.Tensor, shape: (num_seq, num_steps, dim_a),
            Batch of projected manifold latent factors (outputs of encoder; nonlinear manifold embedding step)
        - u: torch.Tensor, shape: (num_seq, num_steps, dim_u),
            Batch of control input vectors
        - mask: torch.Tensor, shape: (num_seq, num_steps, 1),
            Mask input which shows whether observations at each timestep exists (1) or are missing (0)
        - do_smoothing: bool, Whether to run RTS Smoothing or not

        Returns:
        ------------
        - mu_pred_all: torch.Tensor, shape: (num_seq, num_steps, dim_x),
            Dynamic latent factor predictions (t+1|t) where first index of the second dimension has x_{1|0}
        - mu_t_all: torch.Tensor, shape: (num_seq, num_steps, dim_x),
            Dynamic latent factor filtered estimates (t|t) where first index of the second dimension has x_{0|0}
        - mu_back_all: torch.Tensor, shape: (num_seq, num_steps, dim_x),
            Dynamic latent factor smoothed estimates (t|T) where first index of the second dimension has x_{0|T}. Ones tensor if do_smoothing is False
        - Lambda_pred_all: torch.Tensor, shape: (num_seq, num_steps, dim_x, dim_x),
            Dynamic latent factor estimation error covariance predictions (t+1|t) where first index of the second dimension has P_{1|0}
        - Lambda_t_all: torch.Tensor, shape: (num_seq, num_steps, dim_x, dim_x),
            Dynamic latent factor estimation error covariance filtered estimates (t|t) where first index of the second dimension has P_{0|0}
        - Lambda_back_all: torch.Tensor, shape: (num_seq, num_steps, dim_x, dim_x),
            Dynamic latent factor estimation error covariance smoothed estimates (t|T) where first index of the second dimension has P_{0|T}. Ones tensor if do_smoothing is False
        '''

        if do_smoothing:
            mu_pred_all, mu_t_all, mu_back_all, Lambda_pred_all, Lambda_t_all, Lambda_back_all = self.smooth(a=a, u=u, mask=mask)
        else:
            mu_pred_all, mu_t_all, Lambda_pred_all, Lambda_t_all = self.filter(a=a, u=u, mask=mask)
            mu_back_all = torch.ones_like(mu_t_all, device=mu_t_all.device)
            Lambda_back_all = torch.ones_like(Lambda_t_all, device=Lambda_t_all.device)

        return mu_pred_all, mu_t_all, mu_back_all, Lambda_pred_all, Lambda_t_all, Lambda_back_all
