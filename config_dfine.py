'''
Copyright (c) 2023 University of Southern California
See full notice in LICENSE.md
Hamidreza Abbaspourazad*, Eray Erturk* and Maryam M. Shanechi
Shanechi Lab, University of Southern California
'''
import os

from yacs.config import CfgNode as CN
import torch

from python_utils import flatten_dict, unflatten_dict
from datetime import date, datetime


# Initialization of default and recommended config
# (except dimensions and hidden layer lists, set them suitable for data to fit)
_config = CN()

# device and seed
_config.device = 'cpu'
_config.seed = int(torch.randint(low=0, high=100000, size=(1,)))

# model
_config.model = CN()
_config.model.hidden_layer_list = [32,32,32] # Hidden layer list where each element is the number of neurons for that hidden layer of DFINE encoder/decoder. Please use [20,20,20,20] for nonlinear manifold simulations. Set to None to remove the manifold, reducing the model to a pure LDM
_config.model.activation = 'tanh' # Activation function used in encoder and decoder layers. Set to None if hidden_layer_list is None.
_config.model.dim_y = None # Dimensionality of neural observations
_config.model.dim_a = None # Dimensionality of manifold latent factor, a choice higher than dim_y (above) may lead to overfitting
_config.model.dim_x = None # Dimensionality of dynamic latent factor, it's recommended to set it same as dim_a (above), please see Extended Data Fig. 8
_config.model.dim_u = None # Dimensionality of control input
_config.model.init_A_scale = 1. # Initialization scale of LDM state transition matrix
_config.model.init_B_scale = 1. # Initialization scale of LDM control-input matrix
_config.model.init_C_scale = 1. # Initialization scale of LDM observation matrix
_config.model.init_D_scale = 1. # Initialization scale of LDM feed-through matrix
_config.model.fit_D_matrix = False # If true, fits a feed-through matrix
_config.model.init_W_scale = 0.5 # Initialization scale of LDM process noise covariance matrix
_config.model.init_R_scale = 0.5 # Initialization scale of LDM observation noise covariance matrix
_config.model.init_cov = 1 # Initialization scale of dynamic latent factor estimation error covariance matrix
_config.model.is_W_trainable = True # Boolean for whether process noise covariance matrix W is learnable or not
_config.model.is_R_trainable = True # Boolean for whether observation noise covariance matrix R is learnable or not
_config.model.ldm_kernel_initializer = 'default' # Initialization type of LDM parameters, see nn.get_kernel_initializer_function for detailed definition and supported types
_config.model.nn_kernel_initializer = 'xavier_normal' # Initialization type of DFINE encoder and decoder parameters, see nn.get_kernel_initializer_function for detailed definition and supported types
_config.model.supervise_behv = False # Boolean for whether to learn a behavior-supervised model or not. It must be set to True if supervised model will be trained.
_config.model.hidden_layer_list_mapper = [20,20,20] # Hidden layer list for the behavior mapper where each element is the number of neurons for that hidden layer of the mapper
_config.model.activation_mapper = 'tanh' # Activation function used in mapper layers
_config.model.which_behv_dims = [0,1,2,3] # List of dimensions of behavior data to be decoded by mapper, check for any dimensionality mismatch
_config.model.behv_from_smooth = True # Boolean for whether to decode behavior from a_smooth
_config.model.save_dir = os.path.join(os.getcwd(), 'results', 'train_logs', date.today().isoformat(), datetime.now().strftime('%H-%M-%S')) # Main save directory for DFINE results, plots and checkpoints
_config.model.save_steps = 10 # Number of steps to save DFINE checkpoints

# loss
_config.loss = CN()
_config.loss.steps_ahead = [1,2,3,4] # List of number of steps ahead for which DFINE is optimized. For unsupervised and supervised versions, default values are [1,2,3,4] and [1,2], respectively.
_config.loss.scale_steps_ahead = [1.,1.,1.,1.] #relative weighting of each step ahead. Can be zero to calculate that step-ahead prediction (for logging) but not include it in the final loss
_config.loss.scale_l2 = 2e-3 # L2 regularization loss scale (we recommend a grid-search for the best value, i.e., a grid of [1e-4, 5e-4, 1e-3, 2e-3]). Please use 0 for nonlinear manifold simulations as it leads to a better performance.
_config.loss.scale_control_loss = 0.
_config.loss.scale_behv_recons = 20 # If _config.model.supervise_behv is True, scale for MSE of behavior reconstruction (We recommend a grid-search for the best value. It should be set to a large value).
_config.loss.scale_spectr_reg_B = 0.
_config.loss.scale_forward_pred = 0. # Loss scale for forward prediction loss (output is predicted solely from the input)
_config.loss.scale_dyn_x_loss = 0.
_config.loss.scale_con_a_loss = 0.


# training
_config.train = CN()
_config.train.batch_size = 32 # Batch size
_config.train.num_epochs = 200 # Number of epochs for which DFINE is trained
_config.train.valid_step = 1 # Number of steps to check validation data performance
_config.train.plot_save_steps = 50 # Number of steps to save training/validation plots
_config.train.print_log_steps = 10 # Number of steps to print training/validation logs

# loading
_config.load = CN()
_config.load.ckpt = -1 # Number of checkpoint to load
_config.load.resume_train = False # Boolean for whether to resume training from the epoch where checkpoint is saved

# learning rate
_config.lr = CN()
_config.lr.scheduler = 'explr' # Learning rate scheduler type, options are explr (StepLR, purely exponential if explr.step_size == 1), cyclic (CyclicLR) or constantlr (constant learning rate, no scheduling)
_config.lr.init = 0.02 # Initial learning rate

# cyclic LR scheduler, check https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CyclicLR.html for details
_config.lr.cyclic = CN()
_config.lr.cyclic.base_lr = 0.005 # Minimum learning rate for cyclic LR scheduler
_config.lr.cyclic.max_lr = 0.02 # Maximum learning rate for cyclic LR scheduler
_config.lr.cyclic.gamma = 1. # Envelope scale for exponential cyclic LR scheduler mode
_config.lr.cyclic.mode = 'triangular' # Mode for cyclic LR scheduler
_config.lr.cyclic.step_size_up = 10 # Number of iterations in the increasing half of the cycle

# exponential LR scheduler, check https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html for details
_config.lr.explr = CN()
_config.lr.explr.gamma = 0.9 # Multiplicative factor of learning rate decay
_config.lr.explr.step_size = 15 # Steps to decay the learning rate, becomes purely exponential if step is 1

# optimizer
_config.optim = CN()
_config.optim.eps = 1e-8 # Epsilon for Adam optimizer
_config.optim.grad_clip = 1. # Gradient clipping norm


def get_default_config():
    '''
    Creates the default config

    Returns:
    ------------
    - config: yacs.config.CfgNode, default DFINE config
    '''

    return _config.clone()



def make_config(savedir_suffix='', **config_kwargs):
    config = get_default_config()

    #ensures timestamp is current, the one from get_default_config() corresponds to the time the module was initially loaded
    config.model.save_dir = os.path.join(os.getcwd(), 'results', 'train_logs', date.today().isoformat(), datetime.now().strftime('%H-%M-%S'))

    #append any other info to the save directory
    config.model.save_dir += savedir_suffix

    config = update_config(config, config_kwargs, new_is_flat=True)
    if config.model.dim_a is None:
        config.model.dim_a = config.model.dim_x
    if config.model.hidden_layer_list is None:
        config.model.activation = None

    return config


def load_config(path, ckpt=None):
    config = get_default_config()

    if not path.endswith('config.yaml'):
        path = os.path.join(path, 'config.yaml')

    config.merge_from_file(path)
    config.load.ckpt = ckpt or config.train.num_epochs
    return config


def make_savedir_suffix(config, **kwargs):
    suffix = (f'_nx={config["model.dim_x"]}'
              f'_na={config["model.dim_a"]}'
              f'_ny={config["model.dim_y"]}'
              f'_nu={config["model.dim_u"]}'
              f'_nh={config["model.activation"]}-{"-".join(map(str,config["model.hidden_layer_list"]))}'
              )
    if kwargs:
        suffix += '_' + '_'.join([f'{key}={val}' for key,val in kwargs.items()])
    return suffix



def update_config(config, new_config, new_is_flat=False):
    '''
    Updates the config

    Parameters:
    ------------
    - config: yacs.config.CfgNode or dict, Config to update
    - new_config: yacs.config.CfgNode or dict, Config with new settings and appropriate keys

    Returns:
    ------------
    - unflattened_config: yacs.config.CfgNode, Config with updated settings
    '''

    # Flatten both configs
    flat_config = flatten_dict(config)
    flat_new_config = flatten_dict(new_config) if not new_is_flat else new_config

    # Update and unflatten the config to return
    missing_keys = flat_new_config.keys() - flat_config.keys()
    if missing_keys:
        raise KeyError(f"Keys not present in default config: {missing_keys}")

    flat_config.update(flat_new_config)
    unflattened_config = CN(unflatten_dict(flat_config))

    return unflattened_config
