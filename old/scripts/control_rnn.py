import math
from importlib import reload
from copy import deepcopy
import os
os.environ['TQDM_DISABLE'] = '1'

import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from plot_utils import plot_parametric, plot_vs_time
from data.RNN import RNN, ReachRNN
from data.SSM import NonlinearStateSpaceModel
import config_dfine
from datasets import DFINEDataset
from trainers.TrainerDFINE import TrainerDFINE
from controller import LQGController
from python_utils import verify_shape
from time_series_utils import generate_input_noise

# torch.set_default_dtype(torch.float64)


def plot_reach_pos_vel(y_seq, v_seq=None, title=''):
    ncols = 1 if v_seq is None else 2
    fig, ax = plt.subplots(1, ncols, figsize=(5*ncols+1, 5), squeeze=False)
    ax = ax.squeeze(0)

    cbar = (v_seq is None)
    plot_parametric(y_seq, ax=ax[0], varname='y', cbar=cbar, title=f'{title}, position')
    ax[0].scatter(*targets.T, c='k', label='$y_{ss}$')
    ax[0].scatter(*est_targets.T, marker='x', c='k', label='$\\widehat{y}_{ss}$')
    ax[0].axis('square')
    ax[0].legend()
    fig.tight_layout()

    if v_seq is not None:
        plot_parametric(v_seq, ax=ax[1], varname='v', title=f'{title}, velocity')
        ax[1].scatter(*targets.T, c='k')
        ax[1].scatter(*est_targets.T, marker='x', c='k')
        ax[1].axis('square')
        fig.tight_layout()


def plot_outputs_2d(y_seq, perturbed_y_seq, controlled_y_seq, estimated_y_seq, targets, est_targets):
    # Plot rnn, perturbed, controlled observations in 2D
    fig, axs = plt.subplots(2,2, sharex='row', sharey='row', figsize=(9,8))
    axs = axs.flatten()
    plot_parametric(y_seq,            ax=axs[0], cbar=False, varname='y', title='Trained')
    plot_parametric(perturbed_y_seq,  ax=axs[1],             varname='y', title='Perturbed')

    plot_parametric(controlled_y_seq, ax=axs[2], cbar=False, varname='y', title='Controlled')
    plot_parametric(estimated_y_seq,  ax=axs[3],             varname='\hat{y}', title='Estimated')

    for ax in axs:
        ax.scatter(*targets.T.detach(), c='k', label='$y_{ss}$')
        ax.axis('square')
    axs[3].scatter(*est_targets.T.detach(), marker='x', c='k', label='$\\widehat{y}_{ss}$')
    axs[3].legend()

    fig.tight_layout()


#%% Data
n_targets = 8
targets = torch.tensor([[torch.cos(x), torch.sin(x)] for x in torch.linspace(0, 2*math.pi*(n_targets-1)/n_targets, n_targets)])

num_steps = 50
s_seq = targets.unsqueeze(1).expand(-1,num_steps,-1) #[b,t,y]

dim_x = 32
dim_s = dim_y = targets.shape[1]
dim_u = dim_x

#%% Network
seed = 1
load_rnn = False
# load_rnn = '/Users/dtyulman/Drive/dfine_ctrl/torchDFINE/data/RNN_x=32_s=2_u=32_y=2.pt'

train_rnn(dim_x, dim_y, dim_u, dim_s, n_train_iters=1000, seed=1, load=False)



#%% Original and perturbed data
x_seq, v_seq, y_seq = run_rnn(rnn, s_seq=s_seq)
perturbed_x_seq, perturbed_v_seq, perturbed_y_seq = run_rnn(perturbed_rnn, s_seq=s_seq)

nrows = 1 if v_seq is None else 2
fig, axs = plt.subplots(nrows, 2, sharex='row', sharey='row', squeeze=False, figsize=(10,4.5*nrows))

plot_parametric(y_seq,           ax=axs[0,0], cbar=False, varname='y', title='Trained (position)')
plot_parametric(perturbed_y_seq, ax=axs[0,1],             varname='y', title='Perturbed (position)')
if v_seq is not None:
    plot_parametric(v_seq,           ax=axs[1,0], cbar=False, varname='v', title='Trained (velocity)')
    plot_parametric(perturbed_v_seq, ax=axs[1,1],             varname='v', title='Perturbed (velocity)')

for ax in axs.flatten():
    ax.scatter(*targets.T, c='k', label='$y_{ss}$')
    ax.axis('square')
fig.tight_layout()


#%% Generate DFINE training data from perturbed RNN
train_dfine_on = 'position'
include_s = True

num_seqs_dfine = 2**16
train_input = {'lo':-0.5, 'hi':0.5, 'levels':2}
u_dfine = generate_input_noise(dim_u, num_steps, num_seqs=num_seqs_dfine, **train_input)
s_dfine = targets.unsqueeze(1).tile(num_seqs_dfine//n_targets, num_steps, 1) #[b,t,y]

with torch.no_grad():
    x_dfine, v_dfine, y_dfine = run_rnn(perturbed_rnn, s_seq=s_dfine, u_seq=u_dfine)
    u_dfine = torch.cat((s_dfine, u_dfine), dim=-1) if include_s else u_dfine

if train_dfine_on == 'velocity':
    train_data = DFINEDataset(y=v_dfine, u=u_dfine)
elif train_dfine_on == 'position':
    train_data = DFINEDataset(y=y_dfine, u=u_dfine)
elif train_dfine_on == 'both':
    y_dfine = torch.cat((y_dfine, v_dfine), dim=-1)
    train_data = DFINEDataset(y=y_dfine, u=u_dfine)
elif train_dfine_on == 'neurons':
    y_dfine = x_dfine
    train_data = DFINEDataset(y=y_dfine, u=u_dfine)
else:
    raise ValueError()

#%% Plot data sample
n_samples = 128
if train_dfine_on == 'velocity':
    plot_parametric(v_dfine[:n_samples], title='Data sample (velocity)', varname='v')
elif train_dfine_on == 'position':
    plot_parametric(y_dfine[:n_samples], title='Data sample (position)', varname='y')
elif train_dfine_on == 'both':
    plot_parametric(y_dfine[:n_samples,:,:2], title='Data sample (position)', varname='y')
    plot_parametric(y_dfine[:n_samples,:,2:], title='Data sample (velocity)', varname='v')



#%% Train DFINE
load_model = False

# x=16, s=T
# load_model = '/Users/dtyulman/Drive/dfine_ctrl/torchDFINE/results/train_logs/2024-07-21/145630_rnn_neurons_u=-0.5_0.5_2'

#x=32, s=T
# load_model = '/Users/dtyulman/Drive/dfine_ctrl/torchDFINE/results/train_logs/2024-07-21/145419_rnn_neurons_u=-0.5_0.5_2'
# load_model = '/Users/dtyulman/Drive/dfine_ctrl/torchDFINE/results/train_logs/2024-09-24/213737_rnn_position_u=-0.5_0.5_2'
#SR
# load_model = '/Users/dtyulman/Drive/dfine_ctrl/torchDFINE/results/train_logs/2024-10-24/160359_rnn_position_u=-0.5_0.5_2_SR=0.1'
# load_model = '/Users/dtyulman/Drive/dfine_ctrl/torchDFINE/results/train_logs/2024-10-24/160403_rnn_position_u=-0.5_0.5_2_SR=0.01'
# load_model = '/Users/dtyulman/Drive/dfine_ctrl/torchDFINE/results/train_logs/2024-10-24/124718_rnn_position_u=-0.5_0.5_2_SR=0.001'
# load_model = '/Users/dtyulman/Drive/dfine_ctrl/torchDFINE/results/train_logs/2024-10-24/112151_rnn_position_u=-0.5_0.5_2_SR=0.0001'

#CR
# load_model = '/Users/dtyulman/Drive/dfine_ctrl/torchDFINE/results/train_logs/2024-10-25/141117_rnn_position_u=-0.5_0.5_2_SR=0.1'
# load_model = '/Users/dtyulman/Drive/dfine_ctrl/torchDFINE/results/train_logs/2024-10-25/141121_rnn_position_u=-0.5_0.5_2_SR=0.01'


#x=32, s=Fs
# load_model = '/Users/dtyulman/Drive/dfine_ctrl/torchDFINE/results/train_logs/2024-07-21/145424_rnn_neurons_u=-0.5_0.5_2'

#x=4, s=T
# load_model = '/Users/dtyulman/Drive/dfine_ctrl/torchDFINE/results/train_logs/2024-07-21/145633_rnn_neurons_u=-0.5_0.5_2'

#x=2, s=T
# load_model = '/Users/dtyulman/Drive/dfine_ctrl/torchDFINE/results/train_logs/2024-07-21/145307_rnn_neurons_u=-0.5_0.5_2'
#x=2, s=F
# load_model = '/Users/dtyulman/Drive/dfine_ctrl/torchDFINE/results/train_logs/2024-07-21/145327_rnn_neurons_u=-0.5_0.5_2'


#model parameters
reload(config_dfine) #ensures timestamp in get_default_config().model.savedir is current
config = config_dfine.get_default_config()
config.model.dim_x = 32
config.model.dim_u = dim_s + dim_u if include_s else dim_u
config.model.dim_a = config.model.dim_x
if train_dfine_on=='both':
    config.model.dim_y = 2*dim_y
elif train_dfine_on=='neurons':
    config.model.dim_y = dim_x
else:
    config.model.dim_y = dim_y

config.model.hidden_layer_list = [20,20] if config.model.dim_x==2 and train_dfine_on=='position' else [30,30,30,30] #

#training parameters
config.train.num_epochs = 10
config.train.batch_size = 64

config.lr.scheduler = 'constantlr'
config.lr.init = 0.001 #default for Adam
config.loss.scale_l2 = 0
config.loss.scale_spectr_reg_B = 0
config.optim.grad_clip = float('inf')

# config.lr.scheduler = 'explr'
# config.lr.init = 0.02

config.model.save_dir = config.model.save_dir + (f'_rnn_{train_dfine_on}'
                                                 f'_Nx={config.model.dim_x}'
                                                 f"_u={'_'.join([str(v) for v in train_input.values()])}"
                                                 )
if load_model:
    config.model.save_dir = load_model
    config.load.ckpt = config.train.num_epochs
    config.load.resume_train = True

trainer = TrainerDFINE(config)

if not load_model:
    train_loader = DataLoader(train_data, batch_size=config.train.batch_size)
    trainer.train(train_loader)

#%% Run control
from controller import LQGController

model = deepcopy(trainer.dfine).requires_grad_(False)

with torch.no_grad():
    perturbed_rnn.init_state(num_seqs=0, num_steps=1)
    plant = deepcopy(perturbed_rnn).requires_grad_(False)
    if train_dfine_on == 'neurons':
        plant.Wy.data = torch.eye(plant.dim_x)
        plant.by.data = torch.zeros(plant.dim_x,1)
        plant.dim_y = plant.dim_x
    if not include_s:
        plant.dim_s = 0

if train_dfine_on == 'both':
    y_ss = torch.cat((targets, torch.zeros_like(targets)), dim=-1)
elif train_dfine_on == 'neurons':
    x_seq, v_seq, y_seq = run_rnn(rnn, s_seq=s_seq)
    y_ss = x_seq[:,-1,:]
else:
    y_ss = targets


controller = LQGController(plant=plant, model=model, u_min=-0.5, u_max=0.5)
outputs = controller.run_control(y_ss=y_ss, z_ss=targets, R=1e-1, horizon=10, num_steps=50, controller='mpc')
# controller = LQGController(plant=plant, model=model)
# outputs = controller.run_control(y_ss=y_ss, z_ss=targets, R=1e-1, num_steps=50)


if train_dfine_on == 'neurons':
    outputs['y_hat_ss'] = (perturbed_rnn.Wy @ outputs['y_hat_ss'].unsqueeze(-1)).squeeze(-1)
    outputs['y_ss'] = (perturbed_rnn.Wy @ outputs['y_ss'].unsqueeze(-1)).squeeze(-1)
    outputs['y_hat'] = (perturbed_rnn.Wy @ outputs['y_hat'].unsqueeze(-1)).squeeze(-1)
    outputs['y'] = (perturbed_rnn.Wy @ outputs['y'].unsqueeze(-1)).squeeze(-1)
    perturbed_rnn.y_seq = outputs['y']


controlled_y_seq = outputs['y']
controlled_v_seq = None
if train_dfine_on == 'velocity':
    controlled_v_seq = controlled_y_seq.clone()
    controlled_y_seq = perturbed_rnn.velocity_to_position(controlled_v_seq)


#%
plot_outputs_2d(y_seq, perturbed_y_seq, controlled_y_seq, outputs['y_hat'], outputs['y_ss'], outputs['y_hat_ss'])

#%%
fig, ax = controller.plot_all(**outputs, seq_num=1)
for i in [0,1,3]: ax[1,i].xaxis.set_tick_params(labelbottom=True)
if controller.u_min > -float('inf') and controller.u_max < float('inf'):
    for a in ax[:,2]: a.set_ylim(controller.u_min-0.1, controller.u_max+0.1)

#%%
fig, ax = plt.subplots()
ax.stem(torch.linalg.svdvals(model.ldm.B))
ax.set_ylabel('$\\sigma_i(B)$')
ax.set_title(f'$\\lambda_{{CR}}={config.loss.scale_spectr_reg_B}$')
ax.set_ylim(None,1)
fig.set_size_inches(4,2.5)
fig.tight_layout()
# fig.savefig(f'/Users/dtyulman/Desktop/cr={config.loss.scale_spectr_reg_B}.pdf', bbox_inches=0, transparent=True)

#%% Numerically optimize control input
# optimize = 'dfine_linear'

# plant = perturbed_rnn.requires_grad_(False)
# model = deepcopy(trainer.dfine).requires_grad_(False)

# num_seqs, num_steps, dim_s = s_seq.shape
# u_seq = torch.zeros(num_seqs, num_steps, dim_u)
# x_seq, v_seq, y_seq = run_rnn(plant, s_seq=s_seq, u_seq=u_seq)

# if optimize=='direct':
#     u_seq.requires_grad_()
#     opt = torch.optim.Adam((u_seq,))
# elif optimize=='linear':
#     ctrl_fn = nn.Linear(dim_y, dim_u)
#     opt = torch.optim.Adam(ctrl_fn.parameters())
# elif optimize=='nonlinear':
#     ctrl_fn = nn.Sequential(nn.Linear(dim_y, 30), nn.ReLU(),
#                                nn.Linear(30, 30), nn.ReLU(),
#                                nn.Linear(30, 30), nn.ReLU(),
#                                nn.Linear(30, dim_u))
#     opt = torch.optim.Adam(ctrl_fn.parameters())
# elif optimize=='dfine_linear':
#     ctrl_fn = nn.Linear(trainer.dfine.ldm.dim_x, dim_u)
#     controller = LQGController(plant=plant, model=model)
#     opt = torch.optim.Adam(ctrl_fn.parameters())
# elif optimize=='dfine_nonlinear':
#     ctrl_fn = nn.Sequential(nn.Linear(trainer.dfine.ldm.dim_x, 30), nn.ReLU(),
#                                nn.Linear(30, 30), nn.ReLU(),
#                                nn.Linear(30, 30), nn.ReLU(),
#                                nn.Linear(30, dim_u))
#     controller = LQGController(plant=plant, model=model)
#     opt = torch.optim.Adam(ctrl_fn.parameters())
# else:
#     raise ValueError()

# #%% optimize
# loss_fn = LQRCost(Q=torch.eye(dim_y), R=0*torch.eye(dim_u))
# n_train_iters = 20000
# for i in range(n_train_iters):
#     if optimize in ['linear', 'nonlinear']:
#         u_seq = ctrl_fn(y_seq.detach())# - s_seq)

#     if optimize in ['dfine_direct', 'dfine_linear', 'dfine_nonlinear']:
#         outputs = controller.run_control(y_ss, num_steps=num_steps, controller=ctrl_fn)
#         y_seq, u_seq = outputs['y'], outputs['u']
#     else:
#         x_seq, v_seq, y_seq = run_rnn(perturbed_rnn, s_seq, u_seq)

#     loss = loss_fn(y_seq, s_seq, u_seq)

#     opt.zero_grad()
#     loss.backward()
#     opt.step()
#     if i%10 == 0:
#         with torch.no_grad():
#             mse_per_target = (s_seq - y_seq).square().mean(dim=(1,2))
#         print(f'iter={i}, loss={loss:.4f}, mse_per_target={mse_per_target.numpy()}')

# #%% plot
# controlled_x_seq, controlled_v_seq, controlled_y_seq = run_rnn(perturbed_rnn, s_seq=s_seq, u_seq=u_seq)
# fig, ax_y = plot_vs_time(controlled_y_seq, targets, varname='y')
# fig, ax_u = plot_vs_time(u_seq, torch.zeros(u_seq.shape[0], u_seq.shape[-1]), varname='u', max_N=4, max_B=4)
# for i, _ax in enumerate(ax_u[0,:]):
#     _ax.set_title(f'$y_{{ss}}=${y_ss[i].numpy().round(1)}')
# fig.tight_layout()

# fig, ax = controller.plot_all(**outputs, seq_num=1)

# fig, ax_p = plot_parametric(controlled_y_seq, varname='y', title='Controlled')
# ax_p.scatter(*targets.T, c='k', label='$y_{ss}$')
# ax_p.scatter(*est_targets.T, marker='x', c='k', label='$\\widehat{y}_{ss}$')
# ax_p.legend()
# # Try:
# # - Do this with the actual LQR cost using x_ss


# #%% Run control on plant that is clone of trained model
# import types
# import data.SSM
# reload(data.SSM)
# from data.SSM import NonlinearStateSpaceModel
# from controller import LQGController

# #extract model parameters
# model = deepcopy(trainer.dfine).requires_grad_(False)
# A = model.ldm.A #[x,x]
# Bs, Bu = model.ldm.B.split([dim_s, dim_u], dim=1) #[x,s],[x,u]
# C = model.ldm.C #[a,x]
# f = model.decoder #(a->y)
# finv = model.encoder #(y->a)
# Qcov,Rcov = model.ldm._get_covariance_matrices() #[x,x],[a,a]

# num_steps = 50
# t_on = -1
# t_off = float('inf')
# estimated_y_seq = torch.empty(8,num_steps,2)
# controlled_y_seq = torch.empty(8,num_steps,2)
# est_targets = torch.empty(8,2)
# for i,y_ss in  enumerate(targets):
#     print(f'Target: {y_ss.numpy()}')

#     b = Bs @ y_ss
#     plant = NonlinearStateSpaceModel(A,b,Bu,C,f)#,Qcov,Rcov)

#     #trick the controller into thinking that this SSM is an RNN
#     plant.__class__.__name__ = plant.__class__.__name__ + 'RNN'
#     plant.dim_s = dim_s
#     def step(self, s=None, u=None):
#         return NonlinearStateSpaceModel.step(self, u)
#     plant.step = types.MethodType(step, plant)

#     #run control
#     controller = LQGController(plant=plant, model=model)
#     # x_hat_ss = controller.estimate_latents(y_ss.unsqueeze(0))[0]
#     # plant_init = {'x0':x_hat_ss}

#     ground_truth_latents = ''
#     outputs = controller.run_control(y_ss=y_ss.unsqueeze(0), plant_init={}, Q=1, R=0.1,
#                                      num_steps=num_steps, t_on=t_on, t_off=t_off,
#                                      ground_truth_latents=ground_truth_latents)
#     estimated_y_seq[i,:,:] = outputs['y_hat']
#     controlled_y_seq[i,:,:] = outputs['y']
#     est_targets[i,:] = outputs['y_hat_ss']

# # % Plot controller outputs vs time
# fig, ax = controller.plot_all(**outputs, x=plant.x_seq, a=plant.a_seq)
# if ground_truth_latents == 'a':
#     fig.suptitle('Ground truth manifold latents a(t)')
# elif ground_truth_latents == 'x':
#     fig.suptitle('Ground truth dynamic latents x(t)')
# elif ground_truth_latents == 'ax':
#     fig.suptitle('Ground truth manifold and dynamic latents x(t) and a(t)')
# fig.tight_layout()
# ax[1,-1].xaxis.set_tick_params(labelbottom=True)

#%% Plot controlled observations in 2D
# plot_reach_pos_vel(controlled_y_seq, controlled_v_seq, title='Controlled')


#%% Difference between bias due to task instruction input and bias in control rule
assert include_s

Bs, Bu = model.ldm.B.split([plant.dim_s, plant.dim_u], dim=1)

N = 30
y_grid_1 = y_grid_2 = torch.linspace(-1.5, 1.5, N)
y_grid = torch.cartesian_prod(y_grid_1, y_grid_2)
control_bias = (plant.Bu @ torch.linalg.pinv(Bu) @ Bs @ y_grid.unsqueeze(-1)).squeeze(-1)
task_bias = (plant.Bs @ y_grid.unsqueeze(-1)).squeeze(-1)

fig, ax = plt.subplots()
diff = (control_bias-task_bias).abs().mean(dim=1).reshape(len(y_grid_1),len(y_grid_2))
im = ax.pcolormesh(y_grid_1, y_grid_2, diff.T, cmap='Reds')

fig.colorbar(im, label='MAE')
ax.scatter(*targets.T.detach(), c='k', label='$y_{ss}$')

ax.set_xlabel('$y_1$')
ax.set_ylabel('$y_2$')
ax.set_title('$W_sy$ vs $W_uB_u^{-1}B_s y$')
