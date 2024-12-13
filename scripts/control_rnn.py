import os
os.environ['TQDM_DISABLE'] = '1'

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import data.RNN as RNN
from datasets import DFINEDataset
from controllers import make_closed_loop, make_controller
from script_utils import get_model
from plot_utils import plot_parametric, plot_vs_time
from time_series_utils import z_score_tensor
from python_utils import convert_to_tensor, approx_indexof

np.set_printoptions(suppress=True)
torch.set_printoptions(sci_mode=False)
VERBOSE = True #global toggle for printing/plotting. Set to false if running on cluster

#%% Make RNN
rnn_config = {'seed': 10,
              'load': False,
              'save': False,
              'rnn_kwargs': {'dim_h': 32, #hidden
                             'dim_z': None, #output, inferred from dataset
                             'dim_s': None, #task input, inferred from dataset
                             'dim_u': 32, #control input
                             'observe': 'h'
                             },

              'dataset_kwargs': {'name': 'reach',
                                 'n_targets': 8,
                                 'num_steps': 50
                                 },

              'train_kwargs': {'epochs': 2000
                              }
              }

_rnn, rnn_train_data = RNN.make_rnn(**rnn_config)
_rnn.requires_grad_(False)
rnn = RNN.make_perturbed_rnn(_rnn, noise_std=0.03)

#%%
if VERBOSE:
    fig, ax = plt.subplots(1,2, sharex='row', sharey='row', figsize=(9, 4.5))
    rnn_train_data.plot_rnn_output(_rnn, ax=ax[0], label=None)
    rnn_train_data.plot_rnn_output(rnn, ax=ax[1], title='Perturbed')
    fig.tight_layout()


#%% Generate DFINE training data
data_config = {'num_seqs': 2**16,
               'num_steps': 50,
               'lo': -0.5,
               'hi': 0.5,
               'levels': 2}

h_dfine, z_dfine, y_dfine, u_dfine = rnn_train_data.generate_DFINE_dataset(rnn, **data_config)
train_data = DFINEDataset(y=y_dfine, u=u_dfine)

#%%
if VERBOSE:
    fig, ax = plot_parametric(z_dfine[:8], varname='z',
                    title=f"RNN output $z(t)$ (train on $y={rnn_config['rnn_kwargs']['observe']}$)")
    ax.axis('square')
#%% Load, init, or train DFINE

#Load
# model_config = {
#     'load_path': '/Users/dtyulman/Drive/dfine_ctrl/torchDFINE/results/train_logs/2024-12-12/13-20-20_rnn_reach_u=-0.5-0.5-2',
#     'ckpt': None
#     }


#Init with ground truth system
# model_config = {
#     'ground_truth': None
#     }


#Train
model_config = {
    'train_data': train_data,
    'config': {
        'model.dim_x': rnn.dim_h,
        'model.dim_a': rnn.dim_h,
        'model.dim_y': rnn.dim_y,
        'model.dim_u': rnn.dim_u,
        'model.hidden_layer_list': [30,30,30,30],
        'model.activation': 'relu',

        'train.plot_save_steps': 2,
        'train.num_epochs': 10,
        'train.batch_size': 64,
        'lr.init': 0.001,
        'lr.scheduler': 'constantlr',
        'loss.scale_l2': 0,
        'optim.grad_clip': float('inf'),
        }
    }

save_str = (f"_rnn_{rnn_config['dataset_kwargs']['name']}"
            f"_u={data_config['lo']}-{data_config['hi']}-{data_config['levels']}"
            f"_y={rnn_config['rnn_kwargs']['observe']}"
            f"_Nx={model_config['config']['model.dim_x']}"
            f"_Ny={model_config['config']['model.dim_y']}"
            f"_Nu={model_config['config']['model.dim_u']}")
model_config['config']['savedir_suffix'] = save_str

config, trainer = get_model(**model_config)
if VERBOSE:
    print(trainer.dfine)

#%% Run control
closed_loop_config = {'ground_truth': '',
                      }


run_config = {'num_steps': 50,
              }

control_config = {'R': 1, #control
                  'Q': 1e6, #state
                  # 'F': 1, #final state
                  # 'T': run_config['num_steps']-1 #TODO: why -1 ??
                  }

controller = make_controller(trainer.dfine, **control_config)
closed_loop = make_closed_loop(plant=rnn,
                               dfine=trainer.dfine,
                               controller=controller,
                               **closed_loop_config)

# actual target for z
z_target = rnn_train_data.targets #[b,z]

with torch.no_grad():
    if rnn_config['rnn_kwargs']['observe'] in ['h', 'neurons']:
        h_target_mode = 'unperturbed'
        # Both of these are cheating, in principle need to infer this, or infer y_target directly
        if h_target_mode == 'unperturbed':
            # the setting of h that produces z_target output in unperturbed rnn. Only works if output matrix _rnn.Wz is not perturbed
            with torch.no_grad():
                _rnn(s_seq=rnn_train_data[:][0])
            h_target = _rnn.h_seq[:,-1,:] #[b,h]
        elif h_target_mode == 'pinv':
            # one possible setting of h that produces z_target. Unique only if Wz is invertible
            h_target = (torch.linalg.pinv(rnn.Wz) @ z_target.unsqueeze(-1)).squeeze(-1) #[h,z]@[b,z,1]->[b,h,1]->[b,h]
        else:
            raise ValueError()

        # effective z_target induced by choice of h_target
        z_target = (rnn.Wz @ h_target.unsqueeze(-1)).squeeze(-1) #[z,h]@[b,h,1]->[b,z,1]->[b,z]

        y_target = rnn.obs_fn(h_target, None)

    elif rnn_config['rnn_kwargs']['observe'] in ['z', 'outputs']:
        y_target = rnn.obs_fn(None, z_target)

aux_inputs = [{'s': rnn_train_data.targets}] #[{'s':[b,z]}]}
closed_loop.run(y_target=y_target, aux_inputs=aux_inputs, **run_config)

#%% Plot model vars over time
closed_loop.model.plot_all(seq_num=1)

#%% Plot output trajectory (z) corresponding to controlled trajectory (h or z)
fig, ax = plot_parametric(closed_loop.plant.z_seq, varname='z', title='Controlled')
ax.scatter(*z_target.T, color='k', label='$z^*$')
ax.scatter(*(closed_loop.plant.Wz @ closed_loop.model.y_hat_target.unsqueeze(-1)).squeeze(-1).T, marker='x', color='k', label='$\widehat{z}^*$')
ax.legend()
ax.axis('square')
