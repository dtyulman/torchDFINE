import os, sys
os.environ['TQDM_DISABLE'] = '1'

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
import torch

import data.SSM as SSM
from datasets import DFINEDataset
from controllers import make_controller
from closed_loop import make_closed_loop
from script_utils import get_model
from plot_utils import plot_parametric, plot_vs_time
from time_series_utils import z_score_tensor
from python_utils import convert_to_tensor, approx_indexof

np.set_printoptions(suppress=True)
torch.set_printoptions(sci_mode=False)
MAKE_PLOTS = False #global toggle for printing/plotting


#%% Uncomment to send this script to the cluster for execution via Slurm
# from run_on_remote import run_me_on_remote
# run_me_on_remote(time='1:00:00', cpus=2, mem=16); MAKE_PLOTS=False #never plot on remote


#%% Make a ground-truth SSM
ssm_config = {'manifold': 'swiss',
              'dim_a': 2,
              'dim_y': 3,
              'A': torch.tensor([[.95,   0.05],
                                 [.05,   0.9]])
              }

manifold = ssm_config.pop('manifold')
ssm = SSM.make_ssm(manifold, **ssm_config)


print(ssm)
print('eigvals:', [e.real.item() if e.imag==0 else e for e in torch.linalg.eigvals(ssm.A)])


#%% Generate DFINE training data
data_config = {'num_seqs': 2**15,
               'num_steps': 50,
               'lo': -0.5,
               'hi': 0.5,
               'levels': 2}

x_train, a_train, y_train, u_train = SSM.generate_dataset(ssm, **data_config)
train_data = DFINEDataset(y_train, u_train)

#%%
if MAKE_PLOTS:
    SSM.plot_data_sample(x=x_train, a=a_train, y=y_train, f=ssm.f, num_seqs=10)

#%% Load, init, or train DFINE

#Load
# model_config = {
#     'load_path': None,
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
        'model.dim_x': ssm.dim_x,
        'model.dim_a': ssm.dim_a,
        'model.dim_y': ssm.dim_y,
        'model.dim_u': ssm.dim_u,
        'model.hidden_layer_list': [20,20,20,20],
        'model.activation': 'relu',

        'train.num_epochs': 100,
        'train.batch_size': 64,
        'lr.scheduler': 'constantlr',
        'loss.scale_l2': 0,

        'loss.steps_ahead': [0,1,2,3,4],
        'loss.scale_steps_ahead': [1.,1.,1.,1.,1.],

        'loss.scale_dyn_x_loss': 0.,
        'loss.scale_con_a_loss': 0.,

        'optim.grad_clip': float('inf'),
        }
    }

save_str = f"_{manifold}_u={data_config['lo']}-{data_config['hi']}-{data_config['levels']}"

cfg = model_config['config']
for k in ('dyn_x', 'con_a'):
    key = f'loss.scale_{k}_loss'
    if (v := cfg[key]) > 0:
        save_str += f"_L{k.replace('_','')}={v}"
        break

if 0 in cfg['loss.steps_ahead']:
    i = cfg['loss.steps_ahead'].index(0)
    if (v := cfg['loss.scale_steps_ahead'][i]) > 0:
        save_str += f"_Lk0={v}"

model_config['config']['savedir_suffix'] = save_str

print(save_str)


config, trainer = get_model(**model_config)
print(trainer.dfine)

sys.exit()
#%% Run control
closed_loop_config = {'ground_truth': '',
                      'suppress_plant_noise': True,
                      }


run_config = {'num_steps': 100,
              }

control_config = {'R': 1, #control
                  'Q': 0, #state
                  'Qf': 1e5, #final state
                  'horizon': run_config['num_steps']-1 #TODO: why -1 ??
                  }

controller = make_controller(trainer.dfine, **control_config)
closed_loop = make_closed_loop(plant=ssm,
                               dfine=trainer.dfine,
                               controller=controller,
                               **closed_loop_config)


x_target = None

# x_target = [3.5, 3.5]
# x_target = [(0, 6*torch.pi, 41), (-10, 10, 41)]

x_target, a_target, y_target = SSM.make_target(ssm, x_target)
closed_loop.run(y_target=y_target, **run_config)


#%% Plot model vars vs time
plot_x_target = [5*torch.pi, 5.]
seq_num = approx_indexof(plot_x_target, x_target)[1]

closed_loop.model.plot_all(seq_num=seq_num)

#%% Plot controlled trajectory on the manifold in 3D
plot_x_target = [5*torch.pi, 5.]
seq_num = approx_indexof(plot_x_target, x_target)[1]

fig, ax = ssm.f.plot_manifold(rlim=(0, 6*torch.pi))
_, ax = plot_parametric(closed_loop.model.y[seq_num,:,:], mode='line', varname='y', ax=ax)

# actual init and target
ax.scatter(*closed_loop.model.y[seq_num,0,:], marker='.', s=400, c='k')
ax.scatter(*closed_loop.model.y_target[seq_num], marker='*', s=300, c='k')

# estimated init and target
ax.scatter(*closed_loop.model.y_hat[seq_num,0,:], marker=MarkerStyle('.', fillstyle='none'), s=400, c='k')
ax.scatter(*closed_loop.model.y_hat_target[seq_num], marker=MarkerStyle('*', fillstyle='none'), s=300, c='k')


#%% Plot error heatmap
seq_num = None
fig, ax, error = SSM.plot_error_heatmap(closed_loop.model.y,
                                        closed_loop.model.y_target,
                                        x_target, varname='y')
if seq_num is not None:
    ax.scatter(*closed_loop.plant.x_seq[seq_num,0,:], marker='.', s=200, c='k')
    ax.scatter(*x_target[seq_num], marker='*', s=150, c='k')
