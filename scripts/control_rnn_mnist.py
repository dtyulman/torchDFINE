from copy import deepcopy
import os
import sys
from pprint import pprint

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import data.RNN as RNN
import data.SSM as SSM
from datasets import DFINEDataset
from controllers import make_controller
from closed_loop import make_closed_loop
from script_utils import get_model, resume_training
from plot_utils import plot_parametric, plot_vs_time, plot_eigvals, plot_high_dim, subplots_square
from time_series_utils import z_score_tensor, compute_control_error, generate_input_noise, generate_interpolated_inputs
from python_utils import convert_to_tensor, WrapperModule, identity, Timer

os.environ['TQDM_DISABLE'] = '1'
np.set_printoptions(suppress=True)
torch.set_printoptions(sci_mode=False)

VERBOSE = False #global toggle for printing/plotting
DEBUG = False

#%% Uncomment to send this script to the cluster for execution via Slurm
# from run_on_remote import run_me_on_remote
# if not DEBUG: run_me_on_remote(); VERBOSE=False

#%% Make RNN (MNIST)
rnn_config = {'seed': 10,
              'load': False,
              'save': False,
              'perturb': False,
              'rnn_kwargs': {'dim_h': 32, #hidden
                             'dim_z': 10, #output, inferred from dataset
                             'dim_s': None, #task input, inferred from dataset
                             'dim_u': None, #control input, equal to task input if control_task_input=True
                             'train_bias': False,
                             'obs_fn': 'h',
                             'f': torch.tanh,
                             'gain': 0.8,
                             'control_task_input': True,
                             'init_h': 'rand_unif'
                             },

              'dataset_kwargs': {'name': 'MNIST',
                                 'num_steps': 50,
                                 },

              'train_kwargs': {'epochs': 2 if not DEBUG else 0,
                               'loss_fn': RNN.TimeAvgCrossEntropyLoss(),
                               'batch_size': 64,
                              }
              }

tag = ''
plant_str = (f"rnn_{rnn_config['dataset_kwargs']['name']}"
             f"{f'_{tag}' if tag else ''}"
             f"_nh={rnn_config['rnn_kwargs']['dim_h']}"
             f"_y={rnn_config['rnn_kwargs']['obs_fn']}")
print(plant_str)


rnn, rnn_train_data = RNN.make_rnn(**rnn_config)
rnn.requires_grad_(False)

with Timer('Evaluating'):
    rnn_test_data = RNN.get_rnn_dataset(name='mnist', num_steps=50, train_or_test='test')
    batch_size = 1000
    for (rnn_data, train_or_test) in [(rnn_train_data, 'train'),(rnn_test_data, 'test')]:
        correct = 0
        for i in range(len(rnn_data)//batch_size):
            inp, tgt = rnn_data[i*batch_size:(i+1)*batch_size]
            _, out = rnn(s_seq=inp)
            correct += (out.argmax(dim=-1)[:,-1] == tgt[:,-1]).sum()
        print(f'avg {train_or_test} acc', (correct/len(rnn_data)).item())

#%%
if VERBOSE:
    if rnn.f.__name__ == 'identity': #if rnn.f == RNN.identity:
        fig, ax = plot_eigvals(rnn.W)
        RNN.effective_ldm_properties(rnn, verbose=True)

#%%
if VERBOSE:
    s_seq, z_tgt_seq = rnn_train_data[:5]
    h_seq, z_seq = rnn(s_seq=s_seq)
    z_tgt_seq = F.one_hot(z_tgt_seq, num_classes=rnn.dim_z).float()

    plot_vs_time(h_seq, varname='h')
    plot_vs_time(z_seq, z_tgt_seq[:,0,:], varname='z')

#%% Generate DFINE training data
data_config = {'num_seqs': 2**18 if not DEBUG else 2**11,
               'num_steps': 50,
               'excitation': 'data' #'noise', 'data'
               # 'lo': 0,
               # 'hi': 1,
               # 'levels': 20,
               # 'include_task_input': False,
               # 'add_task_input_to_noise': False
               }

dfine_data_str = ''
if data_config['excitation'] == 'noise':
    dfine_data_str =f"u={data_config['lo']}-{data_config['hi']}-{data_config['levels']}"
elif data_config['excitation'] == 'data':
    dfine_data_str =f"_u=data"

#%%
u_dfine = torch.empty(data_config['num_seqs'], data_config['num_steps'], rnn.dim_u)
y_dfine = torch.empty(data_config['num_seqs'], data_config['num_steps'], rnn.dim_y)
batch_size = 2**11
_data_config = data_config.copy(); _data_config.pop('num_seqs')
for i in range(data_config['num_seqs']//batch_size):
    print(i)
    if data_config['excitation'] == 'noise':
        u_batch = generate_input_noise(rnn.dim_u, num_seqs=batch_size, **_data_config) #[b,t,u]

    elif data_config['excitation'] == 'data':
        mode = 'linear' if torch.rand(1) > 0.5 else 'hold'
        samples_per_seq = torch.randint(3,6, (batch_size,))
        u_batch = generate_interpolated_inputs(rnn_train_data.data, num_seqs=batch_size,
                                               samples_per_seq=samples_per_seq, num_steps=data_config['num_steps'],
                                               mode=mode)

    u_dfine[i*batch_size:(i+1)*batch_size] = u_batch
    h_dfine, z_dfine = rnn(u_seq=u_batch)
    y_dfine[i*batch_size:(i+1)*batch_size] = rnn.obs_fn(h_dfine, z_dfine) #[b,t,y]

train_data = DFINEDataset(y=y_dfine, u=u_dfine)



#%%
if VERBOSE:
    u_2d = u_dfine.view(u_dfine.shape[0], u_dfine.shape[1], 28, 28)
    for _ in range(10):
        fig, axs = subplots_square(u_dfine.shape[1], rows=5)
        for t,(u,ax) in enumerate(zip(u_2d[torch.randint(0, u_dfine.shape[0], (1,)).item()], axs.flatten())):
            ax.imshow(u)
            ax.set_title(f't={t}')
        [ax.axis('off') for ax in axs.flatten()]
        fig.tight_layout()

#%%
if VERBOSE:
    fig, ax = plot_parametric(z_dfine[:10], varname='z', title=f"RNN output $z(t)$ (train on $y={rnn_config['rnn_kwargs']['obs_fn']}$)")
    ax.axis('square')

#%% Load, init, or train DFINE

#Load
# model_config = {
#     'load_path': '/Users/dtyulman/Drive/dfine_ctrl/torchDFINE/results/train_logs/2025-04-24/16-04-36_rnn_MNIST_mnist_u=0-1-20_y=h_nx=32_ny=32_nu=784',
#     'ckpt': 80
#     }


#Init with ground truth system
# model_config = {
#     'ground_truth': rnn
#     }


#Train
model_config = {
    'train_data': train_data,
    'config': {
        'model.dim_x': 32,
        'model.dim_a': rnn.dim_y,
        'model.dim_y': rnn.dim_y,
        'model.dim_u': rnn.dim_u,
        'model.hidden_layer_list': [64,64],
        'model.activation': 'relu',

        'train.plot_save_steps': 10,
        'train.num_epochs': 100 if not DEBUG else 0,
        'train.batch_size': 64,
        'lr.scheduler': 'constantlr',
        'lr.init': 0.001,
        # 'lr.scheduler': 'explr'
        # lr.explr.gamma = 0.9 # Multiplicative factor of learning rate decay
        # lr.explr.step_size = 15 # Steps to decay the learning rate, becomes purely exponential if step is 1

        'loss.scale_l2': 0.0001,
        'optim.grad_clip': float('inf'),
        }
    }


save_str = (f"_{plant_str}"
            f"_{dfine_data_str}"
            f"_nx={model_config['config']['model.dim_x']}"
            f"_ny={model_config['config']['model.dim_y']}"
            f"_nu={model_config['config']['model.dim_u']}")
model_config['config']['savedir_suffix'] = save_str


#%%
config, trainer = get_model(**model_config)

if 'load_path' in model_config:
    rnn, rnn_train_data = torch.load(os.path.join(model_config['load_path'], 'rnn.pt'))
else:
    torch.save((rnn, rnn_train_data), os.path.join(config.model.save_dir, 'rnn.pt'))

sys.exit()

#%%
if VERBOSE:
    print(trainer.dfine)
    dfine_ldm_properties = SSM.get_ldm_properties(trainer.dfine.ldm.A,
                                                  trainer.dfine.ldm.B,
                                                  trainer.dfine.ldm.C,
                                                  verbose=True,
                                                  return_mats=True)
    pprint(dfine_ldm_properties)


#%% Re-start/continue training
# resume_training(trainer, train_data, additional_epochs=200,
#                 batch_size=model_config['config']['train.batch_size'],
#                 override_save_dir=None, override_lr=0.002)


#%% Set up control
closed_loop_config = {'ground_truth': '',
                      # 'suppress_plant_noise': True,
                      }

run_config = {'num_steps': 50,
              't_on': 0,
              }

# control_config = {'mode': 'MPC',
#                   'Q': 1e6, #state
#                   'horizon': 10,
#                   'u_min': -1,
#                   'u_max': 1
#                   }

control_config = {'mode': 'LQR',
                  'R':1,
                  'Q': 1e6, #state
                    # 'Qf': 1e8, #final state
                    # 'horizon': run_config['num_steps']-run_config['t_on']-1,
                  'penalize_obs': True, #Q ~ C^T @ C if True, else Q ~ I (same for Qf)
                    # 'include_u_ss': False,
                  }

# control_config = {'mode': 'MinE',
#                   'num_steps': run_config['num_steps']-run_config['t_on']-1,
#                   'MPC_horizon': float('inf'),
#                   'include_u_ss': False,
#                   }


# control_config = {'mode': 'const',
#                   # 'const': z_target
#                   }


# Model
dfine = deepcopy(trainer.dfine)

# Plant
plant = rnn

# Controller
controller = make_controller(dfine, **control_config)

# Closed loop system
closed_loop = make_closed_loop(plant=plant, dfine=dfine, controller=controller, **closed_loop_config)

# Estimated target for y, and effective target for z if y_target calculated by first estimating the h_target
num_targets = 1
idx = torch.randint(0,len(rnn_train_data),(num_targets,))
z_target_input = rnn_train_data[idx][0][:,0,:] #[b,z]
y_target, h_target, z_target = RNN.make_y_target(rnn, z_target=z_target_input, h_target_mode='unperturbed', num_steps=rnn_train_data.num_steps)


#%% Get target in x_hat space, project onto set of valid equilibria
x_hat_target = closed_loop.model.estimate_target(y_target)

x_hat_target_proj_eq1 = controller.find_valid_equilibrium(x_hat_target, alg=1) # project it to nearest equilibrium
x_hat_target_proj_eq2 = controller.find_valid_equilibrium(x_hat_target, alg=2) # project it to nearest equilibrium
# # x_hat_target_proj3 = controller.find_valid_equilibrium(x_hat_target, closed_loop=True, alg=None) # project it to nearest equilibrium

x_hat_target_proj_ctrb, ctrb_basis, ctrb_svals = SSM.is_in_controllable_subspace(
                                            x_hat_target, dfine.ldm.A, dfine.ldm.B,
                                            rank=10)

print('diff proj_eq1', (x_hat_target - x_hat_target_proj_eq1).norm(dim=1).numpy())
print('diff proj_eq2', (x_hat_target - x_hat_target_proj_eq2).norm(dim=1).numpy())
print('diff proj_ctrb', (x_hat_target - x_hat_target_proj_ctrb).norm(dim=1).numpy())


#%% Visualize x_targets and their projections to controllable subspace
fig, axs = plot_high_dim(x_hat_target, d=2, label='$\\widehat{x}^*$', varname='\\widehat{x}')
plot_high_dim(x_hat_target_proj_ctrb, axs=axs, label='$proj_{CTRB,'f'{Uc.shape[1]}''}(\\widehat{x}^*)$', marker='x', same_color=True)


#%% Run control with y_target given in observation space
closed_loop.run(y_target=y_target,
                # plant_init='x_hat_target',
                **run_config)
pprint(RNN.compute_control_errors(closed_loop.plant, closed_loop.model))


#%% Run control with x_hat_target given in model's latent space
x_hat_target = x_hat_target_proj_eq1

closed_loop.run(x_hat_target=x_hat_target,
                # plant_init='x_hat_target',
                **run_config)
pprint(RNN.compute_control_errors(closed_loop.plant, closed_loop.model))


#%% Plot model vars over time
plot_vars = ['x','y','u'] if dfine.encoder.__class__.__name__ == 'WrapperModule' and dfine.encoder.f.__name__ == 'identity' else ['x','a','y','u']

x_true = a_true = None
if plant.__class__.__name__ == 'RNN' and plant.f.__name__ == 'identity':
    x_true = closed_loop.plant.h_seq
elif plant.__class__.__name__ == 'NonlinearStateSpaceModel':
    x_true = closed_loop.plant.x_seq
    a_true = closed_loop.plant.a_seq

seq_num = 0
fig, axs = closed_loop.model.plot_all(seq_num=seq_num, x=x_true, a=a_true, plot_vars=plot_vars)
# if run_config['num_steps'] > data_config['num_steps']:
#     for ax in axs[:,1]:
#         ax.axvline(data_config['num_steps'], color='k', ls='--')

# pm=1
# for i in range(len(axs[:,0])):
#     axs[i,0].set_ylim(closed_loop.model.x_hat_target[seq_num,i]-pm, closed_loop.model.x_hat_target[seq_num,i]+pm)
#     axs[i,1].set_ylim(closed_loop.model.y_hat_target[seq_num,i]-pm, closed_loop.model.y_hat_target[seq_num,i]+pm)


#%% Visualize init, target, final x
fig, axs = plot_high_dim(x_hat_target, d=2, label='$\\widehat{x}^*$', varname='\\widehat{x}', marker='x')
plot_high_dim(closed_loop.model.x_hat[:,-1,:], axs=axs, label='$\\widehat{x}(T)$', s=10, same_color=True)
plot_high_dim(closed_loop.model.x_hat[:, 0,:], axs=axs, label='$\\widehat{x}(0)$', s=10)


#%% Plot output trajectory (z) corresponding to controlled trajectory (h or z)
# plot_line = 'solid'
plot_line = None
fig, ax = RNN.plot_outputs_2d(closed_loop.model, closed_loop.plant, line=plot_line, overlaid=False, sharexy=True, check_z_seq=isinstance(closed_loop.plant, RNN.RNN))

#%%
max_rank = 2
results = []  # To store results for each rank
x_hat_target = closed_loop.model.estimate_target(y_target)

for rank in range(max_rank+1):
    print('rank', rank)
    if rank == 0:
        x_hat_target_proj = controller.find_valid_equilibrium(x_hat_target, alg=2)  # Project to nearest equilibrium
    else:
        x_hat_target_proj, Uc = SSM.is_in_controllable_subspace(
            x_hat_target, dfine.ldm.A, dfine.ldm.B, rank=rank)

    closed_loop.run(x_hat_target=x_hat_target_proj, **run_config)
    control_errors = RNN.compute_control_errors(closed_loop.plant, closed_loop.model)

    print('proj_orth:', (x_hat_target - x_hat_target_proj).norm(dim=1))
    print('control err:')
    pprint(control_errors)

    # Store data for later plotting
    results.append({
        "rank": rank,
        "z_hat": rnn.compute_output(closed_loop.model.y_hat),
        "z_hat_*": rnn.compute_output(closed_loop.model.y_hat_target),
        "control_errors": control_errors,
    })

#%%
from python_utils import int_sqrt
r, c = int_sqrt(max_rank+1)
fig, axs = plt.subplots(r, c, squeeze=False)
for result, ax in zip(results, axs.flatten()):
    ax.scatter(*result["z_hat"][:,0,:].T, color='b', label='$\\widehat{z}(0)$')
    ax.scatter(*result["z_hat"][:,-1,:].T, color='r', label='$\\widehat{z}(T)$')
    ax.scatter(*result["z_hat_*"].T, marker='x', color='k', label='$\\widehat{z}^*$')
    ax.scatter(*rnn.compute_output(y_target).T, marker='x', color='grey', label='$\\widehat{z}^*_{original}$')

    ax.set_title('equilibrium' if result["rank"] == 0 else f'controllable (rank {result["rank"]})')
fig.suptitle('Proj target onto subspace...')
[ax.set_xlabel('$z_1$') for ax in axs[-1, :]]
[ax.set_ylabel('$z_2$') for ax in axs[:, 0]]
axs[0,0].legend()
fig.tight_layout()

#%%
ranks = [result["rank"] for result in results]
error_keys = results[0]["control_errors"].keys()  # Keys from control_errors dictionary

fig, ax = plt.subplots(3,1)
for i,key in enumerate(error_keys):
    errors = [result["control_errors"][key] for result in results]
    ax[i].plot(ranks, errors)

    ax[-1].set_xlabel('Rank')
    ax[i].set_ylabel(f'Control Error: {key}')
    ax[i].grid(True)
ax[0].legend()
