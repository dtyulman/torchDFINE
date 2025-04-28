'''
Copyright (c) 2023 University of Southern California
See full notice in LICENSE.md
Hamidreza Abbaspourazad*, Eray Erturk* and Maryam M. Shanechi
Shanechi Lab, University of Southern California
'''

import torch
from torch import nn
from torch.nn import functional as F

from modules.LDM import LDM
from modules.MLP import MLP
from nn import get_kernel_initializer_function, compute_mse, get_activation_function
from python_utils import WrapperModule, identity

class DFINE(nn.Module):
    '''
    DFINE (Dynamical Flexible Inference for Nonlinear Embeddings) Model.

    DFINE is a novel neural network model of neural population activity with the ability to perform
    flexible inference while modeling the nonlinear latent manifold structure and linear temporal dynamics.
    To model neural population activity, two sets of latent factors are defined: the dynamic latent factors
    which characterize the linear temporal dynamics on a nonlinear manifold, and the manifold latent factors
    which describe this low-dimensional manifold that is embedded in the high-dimensional neural population activity space.
    These two separate sets of latent factors together enable all the above flexible inference properties
    by allowing for Kalman filtering on the manifold while also capturing embedding nonlinearities.
    Here are some mathematical notations used in this repository:
    - y: The high dimensional neural population activity, (num_seq, num_steps, dim_y). It must be Gaussian distributed, e.g., Gaussian-smoothed firing rates, or LFP, ECoG, EEG
    - a: The manifold latent factors, (num_seq, num_steps, dim_a).
    - x: The dynamic latent factors, (num_seq, num_steps, dim_x).


    * Please note that DFINE can perform learning and inference both for continuous data or trial-based data or segmented continuous data. In the case of continuous data,
    num_seq and batch_size can be set to 1, and we let the model be optimized from the long time-series (this is basically gradient descent and not batch-based gradient descent).
    In case of trial-based data, we can just pass the 3D tensor as the shape (num_seq, num_steps, dim_y) suggests. In case of segmented continuous data,
    num_seq can be the number of segments and DFINE provides both per-segment and concatenated inference at the end for the user's convenience. In the concatenated inference,
    the assumption is the concatenation of segments form a continuous time-series (single time-series with batch size of 1).
    '''

    def __init__(self, config):
        '''
        Initializer for an DFINE object. Note that DFINE is a subclass of torch.nn.Module.

        Parameters:
        ------------

        - config: yacs.config.CfgNode, yacs config which contains all hyperparameters required to create the DFINE model
                                       Please see config_dfine.py for the hyperparameters, their default values and definitions.
        '''

        super(DFINE, self).__init__()

        # Get the config and dimension parameters
        self.config = config

        # Set the seed, seed is by default set to a random integer, see config_dfine.py
        torch.manual_seed(self.config.seed)

        # Set the factor dimensions and loss scales
        self._set_dims_and_scales()

        # Initialize LDM parameters
        A, B, C, D, W_log_diag, R_log_diag, mu_0, Lambda_0 = self._init_ldm_parameters()

        # Initialize the LDM
        self.ldm = LDM(dim_x=self.dim_x, dim_u=self.dim_u, dim_a=self.dim_a,
                       A=A, B=B, C=C, D=D,
                       W_log_diag=W_log_diag, R_log_diag=R_log_diag,
                       mu_0=mu_0, Lambda_0=Lambda_0,
                       is_W_trainable=self.config.model.is_W_trainable,
                       is_R_trainable=self.config.model.is_R_trainable,
                       fit_D_matrix=self.config.model.fit_D_matrix)


        # Initialize encoder and decoder(s)
        if self.config.model.hidden_layer_list is None:
            # Make these a passthrough, thus setting the manifold latent to be equal to the observation and reducing the model to a simple LDM
            assert self.dim_y == self.dim_a, 'Manifold latent and observation dimensions must be equal if not using autoencoder'
            assert self.config.model.activation is None, 'Do not provide activation if not using autoencoder'
            self.encoder = self.decoder = WrapperModule(identity)

        else:
            # Initialize the autoencoder
            self.encoder = self._get_MLP(input_dim=self.dim_y,
                                         output_dim=self.dim_a,
                                         layer_list=self.config.model.hidden_layer_list,
                                         activation_str=self.config.model.activation)

            self.decoder = self._get_MLP(input_dim=self.dim_a,
                                         output_dim=self.dim_y,
                                         layer_list=self.config.model.hidden_layer_list[::-1],
                                         activation_str=self.config.model.activation)

        # If asked to train supervised model, get behavior mapper
        if self.config.model.supervise_behv:
            self.mapper = self._get_MLP(input_dim=self.dim_a,
                                        output_dim=self.dim_behv,
                                        layer_list=self.config.model.hidden_layer_list_mapper,
                                        activation_str=self.config.model.activation_mapper)


    def _set_dims_and_scales(self):
        '''
        Sets the observation (y), manifold latent factor (a) and dynamic latent factor (x)
        (and behavior data dimension if supervised model is to be trained) dimensions,
        as well as behavior reconstruction loss and regularization loss scales from config.
        '''

        # Set the dimensions
        self.dim_y = self.config.model.dim_y
        self.dim_a = self.config.model.dim_a
        self.dim_u = self.config.model.dim_u
        self.dim_x = self.config.model.dim_x

        if self.config.model.supervise_behv:
            self.dim_behv = len(self.config.model.which_behv_dims)

        # Set the loss scales for behavior component and for the regularization
        if self.config.model.supervise_behv:
            self.scale_behv_recons = self.config.loss.scale_behv_recons
        self.scale_l2 = self.config.loss.scale_l2
        self.scale_control_loss = self.config.loss.scale_control_loss
        self.scale_spectr_reg_B = self.config.loss.scale_spectr_reg_B

        assert len(self.config.loss.steps_ahead) == len(self.config.loss.scale_steps_ahead)



    def _get_MLP(self, input_dim, output_dim, layer_list, activation_str='tanh'):
        '''
        Creates an MLP object

        Parameters:
        ------------
        - input_dim: int, Dimensionality of the input to the MLP network
        - output_dim: int, Dimensionality of the output of the MLP network
        - layer_list: list, List of number of neurons in each hidden layer
        - activation_str: str, Activation function's name, 'tanh' by default

        Returns:
        ------------
        - mlp_network: an instance of MLP class with desired architecture
        '''

        activation_fn = get_activation_function(activation_str)
        kernel_initializer_fn = get_kernel_initializer_function(self.config.model.nn_kernel_initializer)

        mlp_network = MLP(input_dim=input_dim,
                          output_dim=output_dim,
                          layer_list=layer_list,
                          activation_fn=activation_fn,
                          kernel_initializer_fn=kernel_initializer_fn
                          )
        return mlp_network


    def _init_ldm_parameters(self):
        '''
        Initializes the LDM Module parameters

        Returns:
        ------------
        - A: torch.Tensor, shape: (self.dim_x, self.dim_x), State transition matrix of LDM
        - C: torch.Tensor, shape: (self.dim_a, self.dim_x), Observation matrix of LDM
        - W_log_diag: torch.Tensor, shape: (self.dim_x, ), Log-diagonal of dynamics noise covariance matrix (W, therefore it is diagonal and PSD)
        - R_log_diag: torch.Tensor, shape: (self.dim_a, ), Log-diagonal of observation noise covariance matrix  (R, therefore it is diagonal and PSD)
        - mu_0: torch.Tensor, shape: (self.dim_x, ), Dynamic latent factor prediction initial condition (x_{0|-1}) for Kalman filtering
        - Lambda_0: torch.Tensor, shape: (self.dim_x, self.dim_x), Dynamic latent factor estimate error covariance initial condition (P_{0|-1}) for Kalman filtering

        * We learn the log-diagonal of matrix W and R to satisfy the PSD constraint for cov matrices. Diagnoal W and R are used for the stability of learning
        similar to prior latent LDM works, see (Kao et al., Nature Communications, 2015) & (Abbaspourazad et al., IEEE TNSRE, 2019) for further info
        '''

        kernel_initializer_fn = get_kernel_initializer_function(self.config.model.ldm_kernel_initializer)
        A = kernel_initializer_fn(self.config.model.init_A_scale * torch.eye(self.dim_x))
        B = kernel_initializer_fn(self.config.model.init_B_scale * torch.eye(self.dim_x, self.dim_u))
        C = kernel_initializer_fn(self.config.model.init_C_scale * torch.randn(self.dim_a, self.dim_x))

        # If fit_D_matrix flag is false, D will be initially set to zero and also will not be updated with gradient descent
        if self.config.model.fit_D_matrix:
            D = kernel_initializer_fn(self.config.model.init_D_scale * torch.randn(self.dim_a, self.dim_u))
        else:
            D = torch.zeros(self.dim_a, self.dim_u)


        W_log_diag = torch.log(kernel_initializer_fn(torch.diag(self.config.model.init_W_scale * torch.eye(self.dim_x))))
        R_log_diag = torch.log(kernel_initializer_fn(torch.diag(self.config.model.init_R_scale * torch.eye(self.dim_a))))

        mu_0 = kernel_initializer_fn(torch.zeros(self.dim_x))
        Lambda_0 = kernel_initializer_fn(self.config.model.init_cov * torch.eye(self.dim_x))

        return A, B, C, D, W_log_diag, R_log_diag, mu_0, Lambda_0


    def forward(self, y, u, mask=None):
        '''
        Forward pass for DFINE Model

        Parameters:
        ------------
        - y: torch.Tensor, shape: (num_seq, num_steps, dim_y), High-dimensional neural observations
        - u: torch.Tensor, shape: (num_seq, num_steps, dim_u), Control input vectors
        - mask: torch.Tensor, shape: (num_seq, num_steps, 1),
            Mask input which shows whether observations at each timestep exist (1) or are missing (0)

        Returns:
        ------------
        - model_vars: dict, Dictionary which contains learned parameters, inferrred latents, predictions and reconstructions. Keys are:
            - a_hat: torch.Tensor, shape: (num_seq, num_steps, dim_a), Batch of projected manifold latent factors.
            - a_pred: torch.Tensor, shape: (num_seq, num_steps-1, dim_a), Batch of predicted estimates of manifold latent factors (last index of the second dimension is removed)
            - a_filter: torch.Tensor, shape: (num_seq, num_steps, dim_a), Batch of filtered estimates of manifold latent factors
            - a_smooth: torch.Tensor, shape: (num_seq, num_steps, dim_a), Batch of smoothed estimates of manifold latent factors
            - x_pred: torch.Tensor, shape: (num_seq, num_steps-1, dim_x), Batch of predicted estimates of dynamic latent factors
            - x_filter: torch.Tensor, shape: (num_seq, num_steps, dim_x), Batch of filtered estimates of dynamic latent factors
            - x_smooth: torch.Tensor, shape: (num_seq, num_steps, dim_x), Batch of smoothed estimates of dynamic latent factors
            - Lambda_pred: torch.Tensor, shape: (num_seq, num_steps-1, dim_x, dim_x), Batch of predicted estimates of dynamic latent factor estimation error covariance
            - Lambda_filter: torch.Tensor, shape: (num_seq, num_steps, dim_x, dim_x), Batch of filtered estimates of dynamic latent factor estimation error covariance
            - Lambda_smooth: torch.Tensor, shape: (num_seq, num_steps, dim_x, dim_x), Batch of smoothed estimates of dynamic latent factor estimation error covariance
            - y_hat: torch.Tensor, shape: (num_seq, num_steps, dim_y), Batch of projected estimates of neural observations
            - y_pred: torch.Tensor, shape: (num_seq, num_steps-1, dim_y), Batch of predicted estimates of neural observations
            - y_filter: torch.Tensor, shape: (num_seq, num_steps, dim_y), Batch of filtered estimates of neural observations
            - y_smooth: torch.Tensor, shape: (num_seq, num_steps, dim_y), Batch of smoothed estimates of neural observations
            - A: torch.Tensor, shape: (num_seq, num_steps, dim_x, dim_x), Repeated (tile) state transition matrix of LDM, same for each time-step in the 2nd axis
            - C: torch.Tensor, shape: (num_seq, num_steps, dim_y, dim_x), Repeated (tile) observation matrix of LDM, same for each time-step in the 2nd axis
            - behv_hat: torch.Tensor, shape: (num_seq, num_steps, dim_behv), Batch of reconstructed behavior. None if unsupervised model is trained

        * Terminology definition:
            projected: noisy estimations of manifold latent factors after nonlinear manifold embedding via encoder
            predicted: one-step ahead predicted estimations (t+1|t), the first and last time indices are (1|0) and (T|T-1)
            filtered: causal estimations (t|t)
            smoothed: non-causal estimations (t|T)
        '''

        # Get the dimensions from y
        num_seq, num_steps, _ = y.shape

        # Create the mask if it's None
        if mask is None:
            mask = torch.ones(y.shape[:-1]).unsqueeze(dim=-1)

        # Get the encoded low-dimensional manifold factors (project via nonlinear manifold embedding)
        a_hat = self.encoder(y)

        # Run LDM to infer filtered and smoothed dynamic latent factors
        x_pred, x_filter, x_smooth, Lambda_pred, Lambda_filter, Lambda_smooth = self.ldm(a=a_hat, u=u, mask=mask, do_smoothing=True)
        a_pred = (self.ldm.C @ x_pred.unsqueeze(dim=-1)).squeeze(dim=-1) #  (num_seq, num_steps, dim_a, dim_x) x (num_seq, num_steps, dim_x, 1) --> (num_seq, num_steps, dim_a)
        a_filter = (self.ldm.C @ x_filter.unsqueeze(dim=-1)).squeeze(dim=-1) #  (num_seq, num_steps, dim_a, dim_x) x (num_seq, num_steps, dim_x, 1) --> (num_seq, num_steps, dim_a)
        a_smooth = (self.ldm.C @ x_smooth.unsqueeze(dim=-1)).squeeze(dim=-1) #  (num_seq, num_steps, dim_a, dim_x) x (num_seq, num_steps, dim_x, 1) --> (num_seq, num_steps, dim_a)

        # Remove the last timestep of predictions since it's T+1|T, which is not of interest to us
        x_pred = x_pred[:, :-1, :]
        Lambda_pred = Lambda_pred[:, :-1, :, :]
        a_pred = a_pred[:, :-1, :]

        if self.config.model.fit_D_matrix:
            a_pred = a_pred + (self.ldm.D @ u[:,1:,...].unsqueeze(dim=-1)).squeeze(dim=-1) #first index of a_pred is a_{1|0}, therefore, it has to be summed with u_{1}
            a_filter = a_filter + (self.ldm.D @ u.unsqueeze(dim=-1)).squeeze(dim=-1)
            a_smooth = a_smooth + (self.ldm.D @ u.unsqueeze(dim=-1)).squeeze(dim=-1)

        # Supervise a_seq or a_smooth to behavior if requested -> behv_hat shape: (num_seq, num_steps, dim_behv)
        if self.config.model.supervise_behv:
            if self.config.model.behv_from_smooth:
                behv_hat = self.mapper(a_smooth.reshape(-1, self.dim_a))
            else:
                behv_hat = self.mapper(a_hat.reshape(-1, self.dim_a))
            behv_hat = behv_hat.reshape(-1, num_steps, self.dim_behv)
        else:
            behv_hat = None

        # Get filtered and smoothed estimates of neural observations
        y_hat = self.decoder(a_hat.reshape(-1, self.dim_a)).reshape(num_seq, -1, self.dim_y)
        y_pred = self.decoder(a_pred.reshape(-1, self.dim_a)).reshape(num_seq, -1, self.dim_y)
        y_filter = self.decoder(a_filter.reshape(-1, self.dim_a)).reshape(num_seq, -1, self.dim_y)
        y_smooth = self.decoder(a_smooth.reshape(-1, self.dim_a)).reshape(num_seq, -1, self.dim_y)

        # Dump inferrred latents, predictions and reconstructions to a dictionary
        model_vars = dict(a_hat=a_hat, a_pred=a_pred, a_filter=a_filter, a_smooth=a_smooth,
                          x_pred=x_pred, x_filter=x_filter, x_smooth=x_smooth,
                          Lambda_pred=Lambda_pred, Lambda_filter=Lambda_filter, Lambda_smooth=Lambda_smooth,
                          y_hat=y_hat, y_pred=y_pred, y_filter=y_filter, y_smooth=y_smooth,
                          behv_hat=behv_hat)
        return model_vars


    def get_k_step_ahead_prediction(self, model_vars, k, u):
        '''
        Performs k-step ahead prediction of manifold latent factors, dynamic latent factors and neural observations.

        Parameters:
        ------------
        - model_vars: dict, Dictionary returned after forward(...) call. See the definition of forward(...) function for information.
            - x_filter: torch.Tensor, shape: (num_seq, num_steps, dim_x), Batch of filtered estimates of dynamic latent factors
            - A: torch.Tensor, shape: (num_seq, num_steps, dim_x, dim_x) or (dim_x, dim_x), State transition matrix of LDM
            - B: torch.Tensor, shape: (num_seq, num_steps, dim_x, dim_u) or (dim_x, dim_u), Control-input matrix of LDM
            - C: torch.Tensor, shape: (num_seq, num_steps, dim_y, dim_x) or (dim_y, dim_x), Observation matrix of LDM
        - k: int, Number of steps ahead for prediction

        Returns:
        ------------
        - y_pred_k: torch.Tensor, shape: (num_seq, num_steps-k, dim_y), Batch of predicted estimates of neural observations,
                                                                           the first index of the second dimension is y_{k|0}
        - a_pred_k: torch.Tensor, shape: (num_seq, num_steps-k, dim_a), Batch of predicted estimates of manifold latent factor,
                                                                        the first index of the second dimension is a_{k|0}
        - x_pred_k: torch.Tensor, shape: (num_seq, num_steps-k, dim_x), Batch of predicted estimates of dynamic latent factor,
                                                                        the first index of the second dimension is x_{k|0}
        '''

        # Check whether provided k value is valid or not
        assert k>0 and isinstance(k, int), 'Number of steps ahead prediction value is invalid or of wrong type, k must be a positive integer!'

        # Extract the required variables from model_vars dictionary
        x_filter = model_vars['x_filter'] #[b,t,x]

        # Get the required dimensions
        num_seq, num_steps, _ = x_filter.shape

        # Here is where k-step ahead prediction is iteratively performed
        x_pred_k = x_filter[:, :-k, ...] #[x_{0|0}, x_{1|1}, ..., x_{(T-k)|(T-k)}], will contain [x_{k|0}, x_{(k+1)|1}, ..., x_{T|(T-k)}]
        for i in range(0, k):
            s = slice(i, num_steps-k+i)
            x_pred_k = (self.ldm.A @ x_pred_k.unsqueeze(dim=-1) + self.ldm.B @ u[:,s].unsqueeze(dim=-1)).squeeze(dim=-1)

            # start: [x_{0|0},   x_{1|1},   ..., x_{(T-k)    |(T-k)}]

            # i=0:   [x_{1|0},   x_{2|1},   ..., x_{(T-k+1)  |(T-k)}]
            #
            #     = [ x_{1|0} = A x_{0|0} + B u_{0},
            #         x_{2|1} = A x_{1|1} + B u_{1},
            #         ...
            #         x_{T-k+1|T-k} = A x_{T-k|T-k} + B u_{T-k} ]


            # i=1:  [x_{2|0},   x_{3|1},   ...,   x_{T-k+2|T-k}]
            #
            #     = [ x_{2|0} = A x_{1|0} + B u_{1},
            #         x_{3|1} = A x_{2|1} + B u_{2},
            #         ...
            #         x_{T-k+2|T-k} = A x_{T-k+1|T-k} + B u_{T-k+1} ]


            # i:    [x_{i+1|0}, x_{i+2  |1}, ...,   x_{(T-k+i+1)|(T-k)}]
            #
            #     = [ x_{i+1|0} = A x_{i|0} + B u_{i},
            #         x_{i+2|1} = A x_{i+1|1} + B u_{i+1},
            #         ...
            #         x_{T-k+i+1|T-k} = A x_{T-k+i|T-k} + B u_{T-k+i} ]


            # i=k-1:   [x_{k|0}, x_{(k+1)|1}, ...,   x_{T|(T-k)}]
            #
            #     = [ x_{k|0} = A x_{k-1|0} + B u_{k-1},
            #         x_{k+1|1} = A x_{k|1} + B u_{k},
            #         ...
            #         x_{T|T-k} = A x_{T-1|T-k} + B u_{T-1} ]


        a_pred_k = (self.ldm.C @ x_pred_k.unsqueeze(dim=-1)).squeeze(dim=-1)
        if self.config.model.fit_D_matrix:
            a_pred_k = a_pred_k + (self.ldm.D @ u[:,k:,...].unsqueeze(dim=-1)).squeeze(dim=-1)

        # After obtaining k-step ahead predicted manifold latent factors, they're decoded to obtain k-step ahead predicted neural observations
        decoder = model_vars['decoder'] if 'decoder' in model_vars else self.decoder
        y_pred_k = decoder(a_pred_k)

        #sanity check
        if k == 1:
            assert (y_pred_k == model_vars['y_pred']).all()
            assert (x_pred_k == model_vars['x_pred']).all()
            assert (a_pred_k == model_vars['a_pred']).all()

        return y_pred_k, a_pred_k, x_pred_k


    def compute_loss(self, y, u, model_vars, mask=None, behv=None, y_hat_cl=None, y_target_cl=None):
        '''
        Computes k-step ahead predicted MSE loss, regularization loss and behavior reconstruction loss
        if supervised model is being trained.

        Parameters:
        ------------
        - y: torch.Tensor, shape: (num_seq, num_steps, dim_y), Batch of high-dimensional neural observations
        - mask: torch.Tensor, shape: (num_seq, num_steps, 1), Mask input which shows whether
                                                              observations at each timestep exists (1) or are missing (0)
                                                              if None it will be set to ones.
        - model_vars: dict, Dictionary returned after forward(...) call. See the definition of forward(...) function for information.
        - behv: torch.tensor, shape: (num_seq, num_steps, dim_behv), Batch of behavior data

        Returns:
        ------------
        - loss: torch.Tensor, shape: (), Loss to optimize, which is sum of k-step-ahead MSE loss, L2 regularization loss and
                                         behavior reconstruction loss if model is supervised
        - loss_dict: dict, Dictionary which has all loss components to log on Tensorboard. Keys are (e.g. for config.loss.steps_ahead = [1, 2]):
            - steps_{k}_mse: torch.Tensor, shape: (), {k}-step ahead predicted masked MSE, k's are determined by config.loss.steps_ahead
            - model_loss: torch.Tensor, shape: (), Negative of sum of all steps_{k}_mse
            - behv_loss: torch.Tensor, shape: (), Behavior reconstruction loss, 0 if model is unsupervised
            - reg_loss: torch.Tensor, shape: (), L2 Regularization loss for DFINE encoder and decoder weights
            - total_loss: torch.Tensor, shape: (), Sum of model_loss, behv_loss and reg_loss
        '''

        # Create the mask if it's None
        if mask is None:
            mask = torch.ones(y.shape[:-1]).unsqueeze(dim=-1)

        # Dump individual loss values for logging or Tensorboard
        loss_dict = dict()

        # Iterate over multiple steps ahead
        k_steps_mse_sum = torch.tensor(0.)
        y_pred_all = {}
        for k, scale_k in zip(self.config.loss.steps_ahead, self.config.loss.scale_steps_ahead):
            #TODO this can be made k times faster by reusing previous k's computation
            y_pred_k, _, _ = self.get_k_step_ahead_prediction(model_vars, k=k, u=u.detach())
            #TODO: if I don't detach u, how will this affect this k-step-ahead computation graph?

            y_pred_all[k] = y_pred_k
            mse_pred = compute_mse(y_flat=y[:, k:, :].reshape(-1, self.dim_y),
                                    y_hat_flat=y_pred_k.reshape(-1, self.dim_y),
                                    mask_flat=mask[:, k:, :].reshape(-1,))
            k_steps_mse_sum += scale_k * mse_pred
            loss_dict[f'steps_{k}_mse'] = mse_pred

        model_loss = k_steps_mse_sum
        loss_dict['model_loss'] = model_loss


        # Get MSE loss for behavior reconstruction, 0 if we dont supervise our model with behavior data
        behv_mse = torch.tensor(0.)
        behv_loss = torch.tensor(0.)
        if self.config.model.supervise_behv:
            behv_mse = compute_mse(y_flat=behv[..., self.config.model.which_behv_dims].reshape(-1, self.dim_behv),
                                   y_hat_flat=model_vars['behv_hat'].reshape(-1, self.dim_behv),
                                   mask_flat=mask.reshape(-1,))
            behv_loss = self.scale_behv_recons * behv_mse
        loss_dict['behv_mse'] = behv_mse
        loss_dict['behv_loss'] = behv_loss


        # L2 regularization loss
        reg_loss = torch.tensor(0.)
        if self.scale_l2 > 0:
            self.scale_l2 * sum([torch.norm(param) for name, param in self.named_parameters() if 'weight' in name])
            loss_dict['reg_loss'] = reg_loss


        # B matrix inverse spectral norm regularization loss
        spectr_reg_B_loss = torch.tensor(0.)
        if self.scale_spectr_reg_B > 0:
            # spectr_reg_B_loss = self.scale_spectr_reg_B / torch.linalg.svd(self.ldm.B)[1].min()
            spectr_reg_B_loss = self.scale_spectr_reg_B * (torch.linalg.cond(self.ldm.B) - 1)
            loss_dict['spectr_reg_B_loss'] = spectr_reg_B_loss


        # Control loss
        control_loss = torch.tensor(0.)
        if y_target_cl is not None:
            #TODO: should I use y_hat from closed-loop, or from open-loop in model_vars?
            # are they identical? --> maybe... torch.allclose==True
            # are their computation graphs? (assuming I don't detach u before putting into dfine.forward)
            # if I don't detach, how will this affect the k-step-ahead computation graph?

            # y_hat_T is fn of u_{1:T},
            # u_t is fn of A,B,C,x_hat_t,
            # x_hat_t is fn of A,B,C,\theta,x_hat_{t-1},a_hat_t
            # a_hat_t is fn of \phi,y_t
            y_hat_T = y_hat_cl[:,-1,:] #model_vars['y_filter'][:,-1,:]  #[b,y]

            control_mse = F.mse_loss(y_hat_T, y_target_cl)
            control_loss = self.scale_control_loss * control_mse
            loss_dict['control_mse'] = control_mse
            loss_dict['control_loss'] = control_loss

        # Final loss
        loss = model_loss + behv_loss + reg_loss + spectr_reg_B_loss + control_loss
        loss_dict['total_loss'] = loss
        return loss, loss_dict, y_pred_all
