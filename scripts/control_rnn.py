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


class LQRCost():
    def __init__(self, Q, R):
        """
        Q: [x,x], state error cost
        R: [u,u], control cost
        """
        self.Q = Q
        self.R = R

    def __call__(self, state, target, control, control_target=0):
        """
        state: [b,t,x]
        target: [b,t,x]
        control: [b,t,u]
        """
        x  = state - target
        u = control - control_target

        xT_Q_x = x.unsqueeze(-2) @ self.Q @ x.unsqueeze(-1) #[b,t,1,x]@[x,x]@[b,t,x,1]->[b,t,1,1]
        uT_R_u = u.unsqueeze(-2) @ self.R @ u.unsqueeze(-1) #[b,t,1,u]@[u,u]@[b,t,u,1]->[b,t,1,1]

        return xT_Q_x.mean() + uT_R_u.mean()


# class Controller(nn.Module):
#     def __init__(self)


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


#%% Data
n_targets = 8
targets = torch.tensor([[torch.cos(x), torch.sin(x)] for x in torch.linspace(0, 2*math.pi*(n_targets-1)/n_targets, n_targets)])

num_steps = 50
s_seq = targets.unsqueeze(1).expand(-1,num_steps,-1) #[b,t,y]

dim_x = 32
dim_s = dim_y = targets.shape[1]
dim_u = dim_x

#%% Network
# load_rnn = False
# load_rnn = '/Users/dtyulman/Drive/dfine_ctrl/torchDFINE/data/ReachRNN_x=100_s=2_u=100_y=2_by=0.pt'
# load_rnn = '/Users/dtyulman/Drive/dfine_ctrl/torchDFINE/data/RNN_x=100_s=2_u=100_y=2.pt'
load_rnn = '/Users/dtyulman/Drive/dfine_ctrl/torchDFINE/data/ReachRNN_x=32_s=2_u=32_y=2.pt'

if load_rnn:
    # Load trained and perturbed RNNs
    print(f'Loading {load_rnn}')
    rnn, perturbed_rnn = torch.load(load_rnn)

else:
    # Train RNN
    # rnn = ReachRNN(dim_x=dim_x, dim_s=dim_s, dim_y=dim_y, dim_u=dim_u)
    rnn = RNN(dim_x=dim_x, dim_s=dim_s, dim_y=dim_y, dim_u=dim_u)
    loss_fn = nn.MSELoss()
    opt = torch.optim.Adam(rnn.parameters())
    n_train_iters = 4000
    for i in range(n_train_iters):
        if isinstance(rnn, ReachRNN):
            x_seq, v_seq, y_seq = rnn(s_seq=s_seq)
        else:
            x_seq, y_seq = rnn(s_seq=s_seq)
        loss = loss_fn(s_seq, y_seq)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if i%100 == 0:
            print(f'iter={i}, loss={loss:.4f}')

    # Perturbed RNN
    with torch.no_grad():
        rnn.init_state(num_seqs=0, num_steps=1)
        perturbed_rnn = deepcopy(rnn)
        perturbed_rnn.W += 0.01*torch.randn_like(perturbed_rnn.W)

    # Optionally save
    save_rnn = False
    suffix = '_by=0' if (rnn.by == 0).all() and not rnn.by.requires_grad() else '' #bias of readout set to zero
    save_rnn = f'{rnn.__class__.__name__}_x={rnn.dim_x}_s={rnn.dim_s}_u={rnn.dim_u}_y={rnn.dim_y}{suffix}.pt'
    if save_rnn:
        save_path = os.path.join(os.getcwd(),'data', save_rnn)
        print(f'Saving to {save_path}')
        torch.save((rnn, perturbed_rnn), save_path)



#%% Original and perturbed data
if isinstance(perturbed_rnn, ReachRNN):
    x_seq,           v_seq,           y_seq           =           rnn(s_seq=s_seq)
    perturbed_x_seq, perturbed_v_seq, perturbed_y_seq = perturbed_rnn(s_seq=s_seq)
else:
    x_seq,           y_seq           =           rnn(s_seq=s_seq)
    perturbed_x_seq, perturbed_y_seq = perturbed_rnn(s_seq=s_seq)

fig, axs = plt.subplots(2,2, sharex='row', sharey='row', figsize=(10,12))
plot_parametric(y_seq,            ax=axs[0,0], cbar=False, varname='y', title='Trained (position)')
plot_parametric(perturbed_y_seq,  ax=axs[0,1], cbar=False, varname='y', title='Perturbed (position)')

if isinstance(perturbed_rnn, ReachRNN):
    plot_parametric(v_seq,            ax=axs[1,0], cbar=False, varname='v', title='Trained (velocity)')
    plot_parametric(perturbed_v_seq,  ax=axs[1,1],             varname='v', title='Perturbed (velocity)')

for ax in axs.flatten():
    ax.scatter(*targets.T, c='k', label='$y_{ss}$')
    ax.axis('square')
fig.tight_layout()


#%% Generate DFINE training data from perturbed RNN
include_s = True

num_seqs_dfine = 2**15

train_input = {'type': 'multilevel_noise', 'lo':-.5, 'hi':.5, 'levels':2}
lo, hi = train_input['lo'], train_input['hi']
levels = train_input['levels']
u_seq_dfine = (hi-lo)/(levels-1) * torch.randint(levels, (num_seqs_dfine, num_steps, dim_u)) + lo

s_seq_dfine = targets.unsqueeze(1).tile(num_seqs_dfine//n_targets, num_steps, 1) #[b,t,y]

with torch.no_grad():
    if isinstance(perturbed_rnn, ReachRNN):
        _, v_dfine, y_dfine = perturbed_rnn(s_seq=s_seq_dfine, u_seq=u_seq_dfine)
    else:
        _, y_dfine = perturbed_rnn(s_seq=s_seq_dfine, u_seq=u_seq_dfine)

u_dfine = torch.cat((s_seq_dfine, u_seq_dfine), dim=-1) if include_s else u_seq_dfine

train_dfine_on = 'position'
if train_dfine_on == 'velocity':
    train_data = DFINEDataset(y=v_dfine, u=u_dfine)
elif train_dfine_on == 'position':
    train_data = DFINEDataset(y=y_dfine, u=u_dfine)
elif train_dfine_on == 'both':
    y_dfine = torch.cat((y_dfine, v_dfine), dim=-1)
    train_data = DFINEDataset(y=y_dfine, u=u_dfine)
else:
    raise ValueError()


#% Plot data sample
n_samples = 32
if train_dfine_on == 'velocity':
    plot_parametric(v_dfine[:n_samples], title='Data sample (velocity)', varname='v')
elif train_dfine_on == 'position':
    plot_parametric(y_dfine[:n_samples], title='Data sample (position)', varname='y')
elif train_dfine_on == 'both':
    plot_parametric(y_dfine[:n_samples,:,:2], title='Data sample (position)', varname='y')
    plot_parametric(y_dfine[:n_samples,:,2:], title='Data sample (velocity)', varname='v')



#%% Train DFINE
load_model = False
# load_model = '/Users/dtyulman/Drive/dfine_ctrl/torchDFINE/results/train_logs/2024-05-08/104917_rnn_u=multilevel_noise_-3_3_10'
# load_model = '/Users/dtyulman/Drive/dfine_ctrl/torchDFINE/results/train_logs/2024-05-08/114241_rnn_u=multilevel_noise_-3_3_10'
# load_model = '/Users/dtyulman/Drive/dfine_ctrl/torchDFINE/results/train_logs/2024-05-15/150231_rnn_u=multilevel_noise_-0.5_0.5_10'

#model parameters
reload(config_dfine) #ensures timestamp in get_default_config().model.savedir is current
config = config_dfine.get_default_config()
config.model.dim_x = 4
config.model.dim_u = dim_s + dim_u if include_s else dim_u
config.model.dim_a = 2
config.model.dim_y = 2*dim_y if train_dfine_on=='both' else dim_y
config.model.hidden_layer_list = [20,20]#[30,30,30,30]
config.model.save_dir = config.model.save_dir + ('_rnn'
                                                 f"_u={'_'.join([str(v) for v in train_input.values()])}")
#training parameters
config.train.num_epochs = 20
config.train.batch_size = 32

config.lr.scheduler = 'constantlr'
config.lr.init = 0.001 #default for Adam
config.loss.scale_l2 = 0
config.optim.grad_clip = float('inf')

# config.lr.scheduler = 'explr'
# config.lr.init = 0.02

if load_model:
    config.model.save_dir = load_model
    config.load.ckpt = 180
    config.load.resume_train = True

trainer = TrainerDFINE(config)

if not load_model:
    train_loader = DataLoader(train_data, batch_size=config.train.batch_size)
    trainer.train(train_loader)


#%% Run control
from controller import LQGController

model = deepcopy(trainer.dfine).requires_grad_(False)

if train_dfine_on == 'both':
    y_ss = torch.cat((targets, torch.zeros_like(targets)), dim=-1)
else:
    y_ss = targets

controller = LQGController(plant=perturbed_rnn, model=model)
outputs = controller.run_control(y_ss=y_ss, R=0.01, num_steps=50)

est_targets = outputs['y_hat_ss']
estimated_y_seq = outputs['y_hat']
controlled_y_seq = outputs['y']
controlled_v_seq = None
if train_dfine_on == 'velocity':
    controlled_v_seq = controlled_y_seq.clone()
    controlled_y_seq = perturbed_rnn.velocity_to_position(controlled_v_seq)

fig, ax = controller.plot_all(**outputs, seq_num=0)
ax[1,-1].xaxis.set_tick_params(labelbottom=True)


#%% Numerically optimize control input
optimize = 'direct_nonlinear'

for p in perturbed_rnn.parameters(): p.requires_grad_(False)

num_seqs, num_steps, dim_s = s_seq.shape
u_seq = torch.zeros(num_seqs, num_steps, dim_u)
# lo, hi,levels = -1, 1, 6
# u_seq = (hi-lo)/(levels-1) * torch.randint(levels, (num_seqs, num_steps, dim_u)) + lo
x_seq, y_seq = perturbed_rnn(s_seq=s_seq, u_seq=u_seq)

if optimize=='direct':
    u_seq.requires_grad_()
    opt = torch.optim.Adam((u_seq,))
elif optimize=='direct_linear':
    controller = nn.Linear(dim_y, dim_u)
    opt = torch.optim.Adam(controller.parameters())
elif optimize=='direct_nonlinear':
    controller = nn.Sequential(nn.Linear(dim_y, 30),
                      nn.ReLU(),
                      nn.Linear(30, 30),
                      nn.ReLU(),
                      nn.Linear(30, 30),
                      nn.ReLU(),
                      nn.Linear(30, dim_u))
    opt = torch.optim.Adam(controller.parameters())

# optimize
loss_fn = LQRCost(Q=torch.eye(dim_y), R=0*torch.eye(dim_u))
n_train_iters = 50000
for i in range(n_train_iters):
    if optimize in ['direct_linear', 'direct_nonlinear']:
        u_seq = controller(y_seq.detach() - s_seq)

    if isinstance(perturbed_rnn, ReachRNN):
        x_seq, v_seq, y_seq = perturbed_rnn(s_seq=s_seq, u_seq=u_seq)
    else:
        x_seq, y_seq = perturbed_rnn(s_seq=s_seq, u_seq=u_seq)
    loss = loss_fn(y_seq, s_seq, u_seq)

    opt.zero_grad()
    loss.backward()
    opt.step()
    if i%100 == 0:
        with torch.no_grad():
            mse_per_target = (s_seq - y_seq).square().mean(dim=(1,2))
        print(f'iter={i}, loss={loss:.4f}, mse_per_target={mse_per_target.numpy()}')

# plot
controlled_x_seq, controlled_y_seq = perturbed_rnn(s_seq=s_seq, u_seq=u_seq)
fig, ax_y = plot_vs_time(controlled_y_seq, targets, varname='y')
fig, ax_u = plot_vs_time(u_seq, torch.zeros(u_seq.shape[0], u_seq.shape[-1]), varname='u', max_N=4, max_B=3)
for i, _ax in enumerate(ax_u[0,:]):
    _ax.set_title(f'$y_{{ss}}=${y_ss[i].numpy().round(1)}')
fig.tight_layout()
plot_parametric(controlled_y_seq, varname='y', title='Controlled')


# Try:
# - Optimal linear fn G of error
# - Optimal u_seq for DFINE to converge to x_hat_ss
# - Optimal G for DFINE


#%% Run control on plant that is clone of trained model
import types
import data.SSM
reload(data.SSM)
from data.SSM import NonlinearStateSpaceModel
from controller import LQGController

#extract model parameters
model = deepcopy(trainer.dfine).requires_grad_(False)
A = model.ldm.A #[x,x]
Bs, Bu = model.ldm.B.split([dim_s, dim_u], dim=1) #[x,s],[x,u]
C = model.ldm.C #[a,x]
f = model.decoder #(a->y)
finv = model.encoder #(y->a)
Qcov,Rcov = model.ldm._get_covariance_matrices() #[x,x],[a,a]

num_steps = 50
t_on = -1
t_off = float('inf')
estimated_y_seq = torch.empty(8,num_steps,2)
controlled_y_seq = torch.empty(8,num_steps,2)
est_targets = torch.empty(8,2)
for i,y_ss in  enumerate(targets):
    print(f'Target: {y_ss.numpy()}')

    b = Bs @ y_ss
    plant = NonlinearStateSpaceModel(A,b,Bu,C,f)#,Qcov,Rcov)

    #trick the controller into thinking that this SSM is an RNN
    plant.__class__.__name__ = plant.__class__.__name__ + 'RNN'
    plant.dim_s = dim_s
    def step(self, x, s=None, u=None):
        return NonlinearStateSpaceModel.step(self, x, u)
    plant.step = types.MethodType(step, plant)

    #run control
    controller = LQGController(plant=plant, model=model)
    # x_hat_ss = controller.estimate_latents(y_ss.unsqueeze(0))[0]
    # plant_init = {'x0':x_hat_ss}

    ground_truth_latents = ''
    outputs = controller.run_control(y_ss=y_ss.unsqueeze(0), plant_init={}, Q=1, R=0.1,
                                     num_steps=num_steps, t_on=t_on, t_off=t_off,
                                     ground_truth_latents=ground_truth_latents)
    estimated_y_seq[i,:,:] = outputs['y_hat']
    controlled_y_seq[i,:,:] = outputs['y']
    est_targets[i,:] = outputs['y_hat_ss']

# % Plot controller outputs vs time
fig, ax = controller.plot_all(**outputs, x=plant.x_seq, a=plant.a_seq)
if ground_truth_latents == 'a':
    fig.suptitle('Ground truth manifold latents a(t)')
elif ground_truth_latents == 'x':
    fig.suptitle('Ground truth dynamic latents x(t)')
elif ground_truth_latents == 'ax':
    fig.suptitle('Ground truth manifold and dynamic latents x(t) and a(t)')
fig.tight_layout()
ax[1,-1].xaxis.set_tick_params(labelbottom=True)

#%% Plot rnn, perturbed, controlled observations in 2D
fig, axs = plt.subplots(2,2, sharex='row', sharey='row', figsize=(8.5,8))
axs = axs.flatten()
plot_parametric(y_seq,            ax=axs[0], cbar=False, varname='y', title='Trained')
plot_parametric(perturbed_y_seq,  ax=axs[1], cbar=False, varname='y', title='Perturbed')
plot_parametric(controlled_y_seq, ax=axs[2], cbar=False, varname='y', title='Controlled')
plot_parametric(estimated_y_seq,  ax=axs[3],             varname='\hat{y}', title='Estimated')

for ax in axs:
    ax.scatter(*targets.T, c='k', label='$y_{ss}$')
    ax.axis('square')
axs[3].scatter(*est_targets.T, marker='x', c='k', label='$\\widehat{y}_{ss}$')
axs[3].legend()

fig.tight_layout()

#%% Plot controlled observations in 2D
plot_reach_pos_vel(controlled_y_seq, controlled_v_seq, title='Controlled')
