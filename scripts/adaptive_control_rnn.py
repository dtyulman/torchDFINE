import os, sys
from copy import deepcopy
from pprint import pprint
os.environ['TQDM_DISABLE'] = '1'

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import data.RNN as RNN
import data.SSM as SSM
from datasets import ControlledDFINEDataset, TargetGenerator, DFINEDataset
from controllers import make_controller
from closed_loop import make_closed_loop
from script_utils import get_model, resume_training
from plot_utils import plot_parametric, plot_vs_time, plot_eigvals, plot_high_dim
from time_series_utils import z_score_tensor, compute_control_error
from python_utils import WrapperModule, identity


np.set_printoptions(suppress=True)
torch.set_printoptions(sci_mode=False)
MAKE_PLOTS = True #global toggle for plotting

#%% Uncomment to send this script to the cluster for execution via Slurm
# from run_on_remote import run_me_on_remote
# run_me_on_remote(time='40:00:00', cpus=2, mem=64); MAKE_PLOTS=False #never plot on remote

#%% Make RNN
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
                             'gain': 0.8,
                             'control_task_input': True,
                             'init_h': 'rand_unif',
                             'dt': 1.,
                             'tau': 1.
                             },

              'dataset_kwargs': {'name': 'reach',
                                 'n_targets': 2**16,
                                 'num_steps': 50,
                                 'spacing': 'uniform'
                                 },

              'train_kwargs': {'epochs': 2,
                               'loss_fn': RNN.MSELossVelocityPenalty(50),
                               'batch_size': 64,
                              }
              }


rnn, rnn_train_data = RNN.make_rnn(**rnn_config)
rnn.requires_grad_(False)
rnn.init_state(num_seqs=0, num_steps=1)
# rnn = RNN.make_perturbed_rnn(rnn.original, noise_std=0.2)

#%%
if MAKE_PLOTS:
    if rnn.f.__name__ == 'identity':
        fig, ax = plot_eigvals(rnn.W)
        RNN.effective_ldm_properties(rnn, verbose=True)


#%%
if MAKE_PLOTS:
    rnn_train_data.num_steps = 200
    n_targets = 10
    fig, ax = RNN.plot_rnn_output(rnn_train_data, rnn, plot='z', n_targets=n_targets)
    fig, ax = RNN.plot_rnn_output(rnn_train_data, rnn, plot='h', h_dims=(0,3), n_targets=n_targets)


#%% Generate DFINE training data

data_config = {'num_seqs': 2**16,
                'num_steps': 50,
                'lo': -0.5,
                'hi': 0.5,
                'levels': 2,
                'include_task_input': False,
                'add_task_input_to_noise': False}
data_config_str = f"OL_u={data_config['lo']}-{data_config['hi']}-{data_config['levels']}"


# data_config_str = 'CL'
# data_config = {}
# data_config['num_seqs'] = 2**16
# data_config['run_config'] = {'num_steps': 50,
#                               't_on': 0}
# data_config['control_config'] = {'mode': 'LQR',
#                                   'R': 1,
#                                   'Qf': 1e2, #final state
#                                   'horizon': data_config['run_config']['num_steps']-data_config['run_config']['t_on']-1,
#                                   'penalize_obs': False, #Q ~ C^T @ C if True, else Q ~ I (same for Qf)
#                                   'include_u_ss': False,
#                                   'u_min': -20,
#                                   'u_max': 20}



#%% Load, init, or train DFINE

#Load
# model_config = {
#     # k-step
#     # 'load_path': '/Users/dtyulman/Drive/dfine_ctrl/torchDFINE/results/train_logs/2025-04-14/20-03-27_rnn_reach_kstep_y=h_nx=32_ny=32_nu=2',
#     # 'ckpt': 17800

#     # control
#     # 'load_path': '/Users/dtyulman/Drive/dfine_ctrl/torchDFINE/results/train_logs/2025-04-14/20-03-50_rnn_reach_ctrl_y=h_nx=32_ny=32_nu=2',
#     # 'ckpt': 18420

#     # k-step + control
#     # 'load_path': '/Users/dtyulman/Drive/dfine_ctrl/torchDFINE/results/train_logs/2025-04-14/20-03-55_rnn_reach_kstep+ctrl_y=h_nx=32_ny=32_nu=2',
#     # 'ckpt': 16340

#     # k-step (open-loop)
#     # 'load_path': '/Users/dtyulman/Drive/dfine_ctrl/torchDFINE/results/train_logs/2025-04-14/18-29-53_rnn_reach_ksa_OL_y=h_nx=32_ny=32_nu=2',
#     # 'ckpt': 710

#     #k-step (linear, open-loop)
#     'load_path': '/Users/dtyulman/Drive/dfine_ctrl/torchDFINE/results/train_logs/2025-05-17/11-47-11_rnn_reach_linear+kstep_OL_u=-0.5-0.5-2_y=h_nx=32_ny=32_nu=2',
#     'ckpt': 100

#     # SID (open-loop)
#     # 'load_path': '/Users/dtyulman/Drive/dfine_ctrl/torchDFINE/results/train_logs/2025-04-27/22-25-17_rnn_reach_psid_u=-0.5-0.5-2_y=h_nx=32_ny=32_nu=2',
#     # 'ckpt': 1
#     }


#Init with ground truth system
# model_config = {
#     'ground_truth': rnn
#     }


#Train
model_config = {
    'train_data': None, #start training manually #TODO: incorporate CL training into get_model?

    'config': {
        'model.dim_x': 32,
        'model.dim_a': rnn.dim_y,
        'model.dim_y': rnn.dim_y,
        'model.dim_u': rnn.dim_u,
        'model.hidden_layer_list': [64,64],
        'model.activation': 'relu',
        'model.init_A_scale': 0.95,

        'model.save_steps': 10,
        'train.plot_save_steps': 10,

        'train.num_epochs': 100,
        'train.batch_size': 64,
        'lr.scheduler': 'constantlr',
        'lr.init': 0.001,
        # 'lr.scheduler': 'explr'
        # lr.explr.gamma = 0.9 # Multiplicative factor of learning rate decay
        # lr.explr.step_size = 15 # Steps to decay the learning rate, becomes purely exponential if step is 1

        'loss.scale_l2': .0001,
        'loss.scale_control_loss': 0.,
        'loss.steps_ahead': [1,2,3,4],
        # 'loss.scale_steps_ahead': [0.,0.,0.,0.],
        'optim.grad_clip': 100.,
        }
    }


#Since with CL one epoch is one step, to make CL have same data size as OL, set:
# num_epochs_CL = num_epochs_OL * num_seqs_OL/batch_size
if data_config_str.startswith('CL'):
    batches_per_epoch_OL = data_config['num_seqs'] // model_config['config']['train.batch_size']
    model_config['config']['train.num_epochs'] *= batches_per_epoch_OL
    model_config['config']['train.plot_save_steps'] *= batches_per_epoch_OL
    model_config['config']['model.save_steps'] *= batches_per_epoch_OL


tag = ['large_']
if 'loss.scale_steps_ahead' not in model_config['config'] or all([ks>0 for ks in model_config['config']['loss.scale_steps_ahead']]):
    tag.append('kstep')
if model_config['config']['loss.scale_control_loss'] > 0:
    tag.append('ctrl')
tag = '+'.join(tag)
print(tag)

tag = f'_{tag}' if tag else ''
save_str = (f"_rnn_{rnn_config['dataset_kwargs']['name']}{tag}"
            f"_{data_config_str}"
            f"_y={rnn_config['rnn_kwargs']['obs_fn']}"
            f"_nx={model_config['config']['model.dim_x']}"
            f"_ny={model_config['config']['model.dim_y']}"
            f"_nu={model_config['config']['model.dim_u']}")
model_config['config']['savedir_suffix'] = save_str


# Instantiate the DFINE model
config, trainer = get_model(**model_config)


#%% Load/save the RNN
if 'load_path' in model_config:
    print('Loading RNN...')
    rnn, rnn_train_data = torch.load(os.path.join(model_config['load_path'], 'rnn.pt'))
    # rnn, rnn_train_data = torch.load('/Users/dtyulman/Drive/dfine_ctrl/torchDFINE/results/train_logs/2025-04-14/18-29-53_rnn_reach_ksa_OL_y=h_nx=32_ny=32_nu=2/rnn.pt')
else:
    print('Saving RNN...')
    torch.save((rnn, rnn_train_data), os.path.join(config.model.save_dir, 'rnn.pt'))



#%% Generate DFINE training data
if data_config_str.startswith('OL'):
    h_dfine, z_dfine, y_dfine, u_dfine = rnn_train_data.generate_DFINE_dataset(rnn, **data_config)
    train_data = DFINEDataset(y=y_dfine, u=u_dfine)

elif data_config_str.startswith('CL'):
    target_generator = TargetGenerator(num_targets=trainer.config.train.batch_size,
                                       rnn=rnn, rnn_train_data=rnn_train_data)
    controller = make_controller(trainer.dfine, clone_mats=False, **data_config['control_config'])
    closed_loop_train = make_closed_loop(plant=rnn, dfine=trainer.dfine, controller=controller,
                                         copy_dfine=False) #need to propagate grads through model
    train_data = ControlledDFINEDataset(closed_loop_train, target_generator,
                                        num_steps=data_config['run_config']['num_steps'])

else:
    raise ValueError("Can't infer CL vs OL dataset")

#%% Train the DFINE model
train_loader = DataLoader(train_data, batch_size=trainer.config.train.batch_size)

try:
    trainer.train(train_loader)
except RuntimeError as e:
    if data_config_str.startswith('CL'):
        #every time there is an error, restart training from latest checkpoint but with tighter bounds on control input
        while True:
            print(e)
            load_config = {
                'load_path': config.model.save_dir,
                'ckpt': max(int(f[:-9]) for f in os.listdir(os.path.join(config.model.save_dir, 'ckpts'))
                            if f.endswith('_ckpt.pth') and f[:-9].isdigit())
                }

            config, trainer = get_model(**load_config)

            if data_config['control_config']['u_min'] < data_config['control_config']['u_max']:
                data_config['control_config']['u_min'] += 1
                data_config['control_config']['u_max'] -= 1
            else:
                break

            controller = make_controller(trainer.dfine, clone_mats=False, **data_config['control_config'])
            closed_loop_train = make_closed_loop(plant=rnn, dfine=trainer.dfine, controller=controller,
                                                 copy_dfine=False)
            train_data = ControlledDFINEDataset(closed_loop_train, target_generator,
                                                num_steps=data_config['run_config']['num_steps'])

            try:
                print(f'Re-starting training, resuming at last checkpoint ({config.load.ckpt})...')
                resume_training(trainer, train_data, start_epoch=config.load.ckpt+1, additional_epochs=config.train.num_epochs-config.load.ckpt)
                break
            except:
                continue


#%%
print(trainer.dfine)
dfine_ldm_properties = SSM.get_ldm_properties(trainer.dfine.ldm.A,
                                              trainer.dfine.ldm.B,
                                              trainer.dfine.ldm.C,
                                              verbose=True,
                                              return_mats=True)
pprint(dfine_ldm_properties)


sys.exit()

#%% Plot k-step plot on "validation" set
data_config = {'num_seqs': 1,
                'num_steps': 50,
                'lo': -0.5,
                'hi': 0.5,
                'levels': 2,
                'include_task_input': False,
                'add_task_input_to_noise': False}
_,_, y_val, u_val = rnn_train_data.generate_DFINE_dataset(rnn, **data_config)
val_data = DFINEDataset(y=y_val, u=u_val)

y,u,_,_ = val_data[0:1]
model_vars = trainer.dfine(y,u)
_,_,y_pred = trainer.dfine.compute_loss(y=y, u=u, model_vars=model_vars)
trainer.create_k_step_ahead_plot(y, y_pred, save_and_close=False)


#%% Set up control

# Model
dfine = deepcopy(trainer.dfine)

# Plant
plant = rnn
rnn.init_h = 'zeros'

# dfine.encoder = dfine.decoder = WrapperModule(identity)
# plant = SSM.make_ssm(dfine)
# plant.obs_fn = RNN.ObservationFunction(obs='h', dim_h=rnn.dim_h) #hack to make the NLSSM look like the RNN to the plot_outputs_2d function
# plant.compute_output = rnn.compute_output

# Controller
run_config = {'num_steps': 50,
              't_on': 0
              }
control_config = {'mode': 'LQR',
                  'R':1,
                  # 'Q':1,
                    'Qf':100, #final state
                    'horizon': run_config['num_steps']-run_config['t_on']-1,
                  'penalize_obs': False, #Q ~ C^T @ C if True, else Q ~ I (same for Qf)
                    # 'include_u_ss': False,
                  }

# control_config = {'mode': 'MinE',
#                   'num_steps': run_config['num_steps']-run_config['t_on']-1,
#                   'MPC_horizon': float('inf'),#run_config['num_steps']-run_config['t_on']-1,
#                   'include_u_ss': False,
#                   }

# control_config = {'mode': 'const',
#                   'const': z_target,
#                   }

controller = make_controller(dfine, **control_config)

# Closed loop system
closed_loop = make_closed_loop(plant=plant,
                               dfine=dfine,
                               controller=controller)#,
                               # suppress_plant_noise=True)

# Actual target for z
# z_target = rnn_train_data.targets[:6] #[b,z]
z_target = torch.tensor([[1,1.],[1,-1],[-1,-1],[-1,1],[0,1],[1,0],[0,-1],[-1,0.]])/2

# Estimated target for y, and effective target for z if y_target calculated by first estimating the h_target
h_target_mode = 'unperturbed'
y_target, h_target, z_target = RNN.make_y_target(rnn, z_target, h_target_mode, num_steps=rnn_train_data.num_steps)


# Get target in x_hat space, project onto set of valid equilibria or controllable subspace
x_hat_target = closed_loop.model.estimate_target(y_target)

x_hat_target_proj_eq1 = controller.find_valid_equilibrium(x_hat_target, alg=1) # project it to nearest equilibrium
x_hat_target_proj_eq2 = controller.find_valid_equilibrium(x_hat_target, alg=2) # project it to nearest equilibrium
# x_hat_target_proj3 = controller.find_valid_equilibrium(x_hat_target, closed_loop=True, alg=None) # project it to nearest equilibrium

x_hat_target_proj_ctrb, ctrb_basis, ctrb_svals = SSM.is_in_controllable_subspace(
                                            x_hat_target, dfine.ldm.A, dfine.ldm.B,
                                            rank=2)

print('diff proj_eq1', (x_hat_target - x_hat_target_proj_eq1).norm(dim=1).numpy())
print('diff proj_eq2', (x_hat_target - x_hat_target_proj_eq2).norm(dim=1).numpy())
print('diff proj_ctrb', (x_hat_target - x_hat_target_proj_ctrb).norm(dim=1).numpy())

#%% Visualize x_targets and their projections to controllable subspace
fig, axs = plot_high_dim(x_hat_target, d=2, label='$\\widehat{x}^*$', varname='\\widehat{x}')
plot_high_dim(x_hat_target_proj_ctrb, axs=axs, label='$proj_{CTRB,'f'{ctrb_basis.shape[1]}''}(\\widehat{x}^*)$', marker='x', same_color=True)


#%% Run control with y_target given in observation space
closed_loop.run(y_target=y_target, **run_config)


#%% Run control with x_hat_target given in model's latent space
x_hat_target = x_hat_target_proj_eq2
closed_loop.run(x_hat_target=x_hat_target, **run_config)

errors = RNN.compute_control_errors(closed_loop.plant, closed_loop.model)
pprint(errors)

#%% Histogram of control errors
fig, axs = plt.subplots(1, len(errors), figsize=(14,3))
for (key,val),ax in zip(errors.items(), axs):
    ax.hist(val, bins=20)
    ax.set_title(f'NMSE({key})')
fig.tight_layout()


#%% Plot model vars over time
plot_vars = ['x','y','u'] if dfine.encoder.__class__.__name__ == 'WrapperModule' and dfine.encoder.f.__name__ == 'identity' else ['x','a','y','u']

fig, axs = closed_loop.model.plot_all(seq_num=0, max_plot_rows=7, plot_vars=plot_vars)
fig.set_size_inches(12, 8.75)


#%% Visualize init, target, final x
fig, axs = plot_high_dim(x_hat_target, d=2, label='$\\widehat{x}^*$', varname='\\widehat{x}', marker='x')
plot_high_dim(closed_loop.model.x_hat[:,-1,:], axs=axs, label='$\\widehat{x}(T)$', s=10, same_color=True)
plot_high_dim(closed_loop.model.x_hat[:, 0,:], axs=axs, label='$\\widehat{x}(0)$', s=10)


#%% Plot output trajectory (z) corresponding to controlled trajectory (h or z)
plot_line = 'solid'
fig, ax = RNN.plot_outputs_2d(closed_loop.model, closed_loop.plant, line=plot_line, overlaid=False, sharexy=True, check_z_seq=isinstance(closed_loop.plant, RNN.RNN))
