from copy import deepcopy
import os
from pprint import pprint
os.environ['TQDM_DISABLE'] = '1'

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import data.RNN as RNN
import data.SSM as SSM
from datasets import DFINEDataset
from controllers import make_controller
from closed_loop import make_closed_loop
from script_utils import get_model, resume_training
from plot_utils import plot_parametric, plot_vs_time, plot_eigvals, plot_high_dim
from time_series_utils import z_score_tensor, compute_control_error, generate_input_noise
from python_utils import convert_to_tensor, WrapperModule, identity

np.set_printoptions(suppress=True)
torch.set_printoptions(sci_mode=False)
VERBOSE = True #global toggle for printing/plotting

# torch.set_default_dtype(torch.float64)
# torch.set_default_dtype(torch.float32)
#%% Make RNN (Reach)
rnn_config = {'seed': 10,
              'load': False,
              'save': False,
              'perturb': False,
              'rnn_kwargs': {'dim_h': 32, #hidden
                             'dim_z': None, #output, inferred from dataset
                             'dim_s': None, #task input, inferred from dataset
                             'dim_u': None, #control input, equal to task input if control_task_input=True
                             'train_bias': False,
                             'obs_fn': 'h',
                             'f': RNN.identity,
                             'gain': 1,
                             'control_task_input': True,
                             'init_h': 'rand_unif'
                             },

              'dataset_kwargs': {'name': 'reach',
                                 'n_targets': 2**16,
                                 'num_steps': 50,
                                 'spacing': 'uniform'
                                 },

              'train_kwargs': {'epochs': 2,
                                # 'loss_fn': RNN.MSELossVelocityPenalty(50),
                               'batch_size': 64,
                              }
              }


rnn, rnn_train_data = RNN.make_rnn(**rnn_config)
rnn.requires_grad_(False)
# rnn = RNN.make_perturbed_rnn(rnn.original, noise_std=0.2)

#%%
if VERBOSE:
    if rnn.f.__name__ == 'identity': #if rnn.f == RNN.identity:
        fig, ax = plot_eigvals(rnn.W)
        RNN.effective_ldm_properties(rnn, verbose=True)

#%%
if VERBOSE:
    rnn_train_data.num_steps = 50
    n_targets = 10
    fig, ax = RNN.plot_rnn_output(rnn_train_data, rnn, plot='z', n_targets=n_targets)
    fig, ax = RNN.plot_rnn_output(rnn_train_data, rnn, plot='h', h_dims=(0,3), n_targets=n_targets)

#%%
# z_target = rnn_train_data.targets[:1000]

h_target = {}
z_target_eff = {}
_, h_target['true'], z_target_eff['true'] = RNN.make_y_target(rnn, z_target, h_target_mode='unperturbed', num_steps=rnn_train_data.num_steps)
_, h_target['pinv'], z_target_eff['pinv'] = RNN.make_y_target(rnn, z_target, h_target_mode='pinv')

# h_target['rand'] = 2*torch.rand_like(h_target['true'])-1

#%%
A, B, _, _ = RNN.effective_ldm_params(rnn)
for key in ['true', 'pinv']:
    is_in_controllable_subspace, proj, U_ctrb = SSM.is_in_controllable_subspace(h_target[key], A, B, verbose=True, return_projection=True)
    h_target[f'{key}_proj'] = convert_to_tensor(proj)
    z_target_eff[f'{key}_proj'] = rnn.compute_output(h_target[f'{key}_proj'])

    print(f'h_target_{key}', is_in_controllable_subspace)

#%%
for key in h_target.keys():
    print(f'h_target_{key} rank=', np.linalg.matrix_rank(h_target[key]))

#%%

fig = plt.figure()
ax = fig.add_subplot()
for key in ['true', 'pinv']:
    sc = ax.scatter(*z_target_eff[key].T, label=key)
    ax.scatter(*z_target_eff[f'{key}_proj'].T, label=f'{key}_proj', color=sc.get_facecolor(), marker='x')
ax.set_xlabel('$z_1$')
ax.set_ylabel('$z_2$')
ax.legend()
ax.axis('square')


#%%
axs = None
for key in ['true', 'pinv']:
    fig, axs = plot_high_dim(h_target[key], d=3, axs=axs, label=key, varname='h')
    plot_high_dim(h_target[f'{key}_proj'], d=3, axs=axs, label=f'{key}_proj', marker='x', varname='h', same_color=True)


#%% Generate DFINE training data
data_config = {'num_seqs': 2**18,
               'num_steps': 50,
               'lo': -0.5,
               'hi': 0.5,
               'levels': 2,
               'include_task_input': False,
               'add_task_input_to_noise': False}

h_dfine, z_dfine, y_dfine, u_dfine = rnn_train_data.generate_DFINE_dataset(rnn, **data_config)
train_data = DFINEDataset(y=y_dfine, u=u_dfine)

#%%
if VERBOSE:
    fig, ax = plot_parametric(z_dfine[:10], varname='z', title=f"RNN output $z(t)$ (train on $y={rnn_config['rnn_kwargs']['obs_fn']}$)")
    ax.axis('square')
#%% Load, init, or train DFINE

#Load
# model_config = {
#     'load_path': '/Users/dtyulman/Drive/dfine_ctrl/torchDFINE/results/train_logs/2024-12-25/20-32-27_rnn_reach_lin_ctrltsk_u=-0.5-0.5-2_y=h_nx=32_ny=32_nu=2',
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
        'train.num_epochs': 100,
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

tag = 'mnist'
tag = f'_{tag}' if tag else ''
save_str = (f"_rnn_{rnn_config['dataset_kwargs']['name']}{tag}"
            f"_u={data_config['lo']}-{data_config['hi']}-{data_config['levels']}"
            f"_y={rnn_config['rnn_kwargs']['obs_fn']}"
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
resume_training(trainer, train_data, additional_epochs=200,
                batch_size=128,
                override_save_dir=None, override_lr=0.002)


#%% Set up control

closed_loop_config = {'ground_truth': '',
                      'suppress_plant_noise': True,
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
                  'R':1e-7,
                    # 'Q': 1, #state
                    'Qf': 1e8, #final state
                    'horizon': run_config['num_steps']-run_config['t_on']-1,
                  'penalize_obs': False, #Q ~ C^T @ C if True, else Q ~ I (same for Qf)
                    # 'include_u_ss': False,
                  }


# control_config = {'mode': 'MinE',
#                   'num_steps': run_config['num_steps']-run_config['t_on']-1,
#                   'MPC_horizon': float('inf'),
#                   'include_u_ss': False,
#                   }


# actual target for z
# z_target = rnn_train_data.targets #[b,z]
z_target = torch.tensor([[1,1.],[1,-1],[-1,-1],[-1,1],[0,1],[1,0],[0,-1],[-1,0.]])/2
# z_target = torch.cartesian_prod(torch.linspace(-1,1,2), torch.linspace(-1,1,2))

# control_config = {'mode': 'const',
#                   # 'const': z_target
#                   }


# Model
dfine = deepcopy(trainer.dfine)
# dfine.encoder = dfine.decoder = WrapperModule(identity)

# Plant
# plant = rnn
plant = SSM.make_ssm(dfine)
plant.obs_fn = RNN.ObservationFunction(rnn, obs='h') #hack to make the NLSSM look like the RNN to the plot_outputs_2d function
plant.compute_output = rnn.compute_output

# Controller
controller = make_controller(dfine, **control_config)

# Closed loop system
closed_loop = make_closed_loop(plant=plant, dfine=dfine, controller=controller, **closed_loop_config)

# Estimated target for y, and effective target for z if y_target calculated by first estimating the h_target
h_target_mode = 'unperturbed'
y_target, h_target, z_target = RNN.make_y_target(rnn, z_target, h_target_mode, num_steps=rnn_train_data.num_steps)

rnn.init_h = 'zeros'
aux_inputs = [{'s': z_target}] if data_config['include_task_input'] else None #[{'s':[b,z]}]}



#% Get target in x_hat space, project onto set of valid equilibria
x_hat_target = closed_loop.model.estimate_target(y_target)

x_hat_target_proj1 = controller.find_valid_equilibrium(x_hat_target, alg=1) # project it to nearest equilibrium
x_hat_target_proj2 = controller.find_valid_equilibrium(x_hat_target, alg=2) # project it to nearest equilibrium
# x_hat_target_proj3 = controller.find_valid_equilibrium(x_hat_target, closed_loop=True, alg=None) # project it to nearest equilibrium

print((x_hat_target - x_hat_target_proj1).norm(dim=1))
print((x_hat_target - x_hat_target_proj2).norm(dim=1))
# print((x_hat_target - x_hat_target_proj3).norm(dim=1))


#% Get target in x_hat space, project onto controllable subspace
x_hat_target_proj, Uc = SSM.is_in_controllable_subspace(
                                            x_hat_target,
                                            dfine.ldm.A, dfine.ldm.B,
                                            rank=2
                                            )

print((x_hat_target - x_hat_target_proj).norm(dim=1))


#%% Visualize x_targets and their projections to controllable subspace
fig, axs = plot_high_dim(x_hat_target, d=2, label='$\\widehat{x}^*$', varname='\\widehat{x}')
plot_high_dim(x_hat_target_proj, axs=axs, label='$proj_{CTRB,'f'{Uc.shape[1]}''}(\\widehat{x}^*)$', marker='x', same_color=True)


#%% Run control with y_target given in observation space
closed_loop.run(y_target=y_target, aux_inputs=aux_inputs,
                # plant_init='x_hat_target',
                **run_config)
pprint(RNN.compute_control_errors(closed_loop.plant, closed_loop.model))


#%% Run control with x_hat_target given in model's latent space
x_hat_target = x_hat_target_proj

closed_loop.run(x_hat_target=x_hat_target, aux_inputs=aux_inputs,
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
