from copy import deepcopy
from collections import defaultdict
import os
os.environ['TQDM_DISABLE'] = '1'

import matplotlib
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from data.SSM import NonlinearStateSpaceModel, SwissRoll
from plot_utils import plot_parametric, plot_vs_time
from datasets import DFINEDataset
from trainers.TrainerDFINE import TrainerDFINE
from config_dfine import get_default_config
from time_series_utils import z_score_tensor
from controller import LQGController

class WrapperModule(torch.nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, *args, **kwargs):
        return self.f(*args, **kwargs)


#%% Make a ground-truth SSM
dim_x = 2
dim_u = 2
dim_a = 2
dim_y = 3

#A = torch.tensor([[.95,   0.05],
#                  [.05,   .9]])
A = torch.tensor([[1,   0.05],
                  [0,   .98]])
f = SwissRoll(dim_y)

Q = 1e-2 * torch.diag(torch.ones(dim_x))  #cf W in DFINE
R = 1e-2 * torch.diag(torch.ones(dim_a))  #cf R in DFINE
S = 2e-3 * torch.diag(torch.ones(dim_y))

ssm = NonlinearStateSpaceModel(A=A, f=f, dim_u=dim_u, dim_a=dim_a)

print(ssm)
#%% Generate data
num_seqs = 1024
num_steps = 200

# train_input = {'type': 'none'}
# train_input = {'type': 'const', 'val': 0.2}
# train_input = {'type': 'gaussian', 'std': 2}
train_input = {'type': 'binary_noise', 'lo': -.5, 'hi': .5}

if train_input['type'] == 'none':
    u = torch.zeros(num_seqs, num_steps, dim_u, dtype=torch.float32)
elif train_input['type'] == 'const':
    u = train_input['val'] * torch.ones(num_seqs, num_steps, dim_u, dtype=torch.float32)
elif train_input['type'] == 'gaussian':
    u = train_input['std'] * torch.randn(num_seqs, num_steps, dim_u, dtype=torch.float32)
elif train_input['type'] == 'binary_noise':
    # lo, hi = torch.randn(num_seqs, 1, dim_u), torch.randn(num_seqs, 1, dim_u)
    lo, hi = train_input['lo'], train_input['hi']
    u = (hi-lo) * torch.bernoulli(0.5*torch.ones(num_seqs,num_steps,dim_u)) + lo
else:
    raise ValueError(f'Invalid train_input type: {train_input["type"]}')

xmin = -10
xmax = 10
x0 = torch.rand(num_seqs, dim_x)*(xmax-xmin) + xmin
x,a,y = ssm(x0=x0, u_seq=u, num_seqs=num_seqs)

#%% Plot data sample
# fig, ax = f.plot_manifold(hlim=(a[:,:,1].min(), a[:,:,1].max()), rlim=(a[:,:,0].min(), a[:,:,0].max()))
ax_x = ax_a = ax_y = None
cb = True
for i in range(10):
    _, ax_x = plot_parametric(x[i], mode='line', ax=ax_x, cbar=cb, varname='x')
    _, ax_a = plot_parametric(a[i], mode='line', ax=ax_a, cbar=cb, varname='a')
    _, ax_y = plot_parametric(y[i], mode='line', ax=ax_y, cbar=cb, varname='y')
    cb = False


#%% Train DFINE
use_ground_truth = False
# load_model = False

# Swiss
# load_model = "/Users/dtyulman/Drive/dfine_ctrl/torchDFINE/results/train_logs/2024-03-15/154837_u={'type': 'binary_noise', 'lo': -0.5, 'hi': 0.5}"
# load_model = '/Users/dtyulman/Drive/dfine_ctrl/torchDFINE/results/train_logs/2024-04-11/164847_u=binary_noise_-0.5_0.5'
load_model = "/Users/dtyulman/Drive/dfine_ctrl/torchDFINE/results/train_logs/2024-03-15/133038_u={'type': 'binary_noise', 'lo': -0.2, 'hi': 0.2}"


config = get_default_config()
config.model.dim_x = dim_x
config.model.dim_u = dim_u
config.model.dim_a = dim_a
config.model.dim_y = dim_y
config.model.save_dir = config.model.save_dir + (f"_u={'_'.join([str(v) for v in train_input.values()])}")
config.model.hidden_layer_list = [20,20,20,20]

config.train.num_epochs = 100
config.train.batch_size = 32
config.lr.scheduler = 'constantlr'
config.loss.scale_l2 = 0

if load_model:
    config.model.save_dir = load_model
    config.load.ckpt = config.train.num_epochs

trainer = TrainerDFINE(config)

if not load_model:
    if use_ground_truth:
        trainer.dfine.ldm.A.data = ssm.A
        trainer.dfine.ldm.B.data = ssm.B
        trainer.dfine.ldm.C.data = ssm.C
        trainer.dfine.ldm.W_log_diag.data = torch.diag(torch.log(ssm.Q_distr.covariance_matrix))
        trainer.dfine.ldm.R_log_diag.data = torch.diag(torch.log(ssm.R_distr.covariance_matrix))
        trainer.dfine.decoder = WrapperModule(ssm.f)
        trainer.dfine.encoder = WrapperModule(ssm.f.inv)
    else:
        train_data = DFINEDataset(y, u)
        train_loader = DataLoader(train_data, batch_size=config.train.batch_size)
        trainer.train(train_loader)


#%% Print model parameters
import numpy as np
np.set_printoptions(suppress=True)
with torch.no_grad():
    print('train_input:', train_input)
    print(trainer.dfine.ldm)


#%% Plot encoder/decoder outputs
#ground truth manifold
rlim=(0,7)#(a[:,:,0].min(), a[:,:,0].max())
hlim=(-7,7)#(a[:,:,1].min(), a[:,:,1].max())
r = torch.linspace(rlim[0], rlim[1], 40)
h = torch.linspace(hlim[0], hlim[1], 15)
a_inputs_gt = torch.cartesian_prod(r, h)
y_samples_gt = ssm.compute_observation(a_inputs_gt, noise=False)

#learned manifold
# rlim=(-8*torch.pi, 8*torch.pi)#(a[:,:,0].min(), a[:,:,0].max())
# hlim=(-5,5)#(a[:,:,1].min(), a[:,:,1].max())
# r = torch.linspace(rlim[0], rlim[1], 400)
# h = torch.linspace(hlim[0], hlim[1], 10)
# a_inputs = torch.cartesian_prod(r, h)
r = torch.linspace(rlim[0], rlim[1], 200)
h = torch.linspace(hlim[0], hlim[1], 60)
a_inputs_ae = torch.cartesian_prod(r, h)
y_samples_ae = ssm.compute_observation(a_inputs_ae, noise=False)
a_inputs = trainer.dfine.encoder(y_samples_ae)
with torch.no_grad():
    y_samples = trainer.dfine.decoder(a_inputs)

#plotting
i = 1
fig = plt.figure(figsize=(7,5))
ax = fig.add_subplot(1,1,1, projection='3d' if dim_y==3 else None)
im1 = ax.scatter(*y_samples_gt.T, marker='.', c=a_inputs_gt[:,i], cmap='coolwarm')
im2 = ax.scatter(*y_samples.T,    marker='.', c=a_inputs_ae[:,i],    cmap='cool')
# fig.colorbar(im1, label=f'SSM $a_{i+1}(t)$')
# fig.colorbar(im2, label=f'DFINE $a_{i+1}(t)$')
fig.colorbar(im1, label=f'SSM $a_{i+1}(t)$')
fig.colorbar(im2, label=f'DFINE reconstruction', ticks=[])

ax.set_title('Decoder $f_{\\theta}(a(t))$ outputs')
ax.set_xlabel('$y_1$')
if dim_y>1: ax.set_ylabel('$y_2$')
if dim_y>2: ax.set_zlabel('$y_3$')


fig.tight_layout()

#Encoder
# y_inputs = torch.rand(10000, dim_y)*100-50
# with torch.no_grad():
#     a_samples = trainer.dfine.encoder(y_inputs)

# ax = fig.add_subplot(1,2,2)
# ax.scatter(y_inputs.norm(dim=1), a_samples, marker='.')
# ax.set_title('Encoder outputs')
# ax.set_ylabel('$a(t)$')
# ax.set_xlabel('$||y(t)||$')

# fig.tight_layout()


#%% Control
model = deepcopy(trainer.dfine).requires_grad_(False)
plant = deepcopy(ssm)
plant.Q_distr = plant.R_distr = plant.S_distr = None

controller = LQGController(plant, model)
num_steps = 50
t_on = 0
t_off = float('inf')
horizon = 50#float('inf')

Q = 0
R = 1
F = 1e10

# Set start and target points. Must be set in plant's latent space `a` or `x` to ensure valid `y`
x_ss_0 = torch.tensor([-3.5,])
x_ss_1 = torch.tensor([-3.5,])
# x_ss_0 = torch.linspace(-4, 4, 30)
# x_ss_1 = torch.linspace(-4, 4, 30)

x_ss = torch.cartesian_prod(x_ss_0, x_ss_1)
a_ss = plant.compute_manifold_latent(x_ss)
y_ss = plant.compute_observation(a_ss)

x0 = torch.tensor([0., 0.], dtype=torch.float32).expand(x_ss.shape)
outputs = controller.run_control(y_ss, {'x0':x0}, num_steps=num_steps, F=F, Q=Q, R=R, t_on=t_on, t_off=t_off, horizon=horizon)

#% Plot controlled dynamics
fig, ax = controller.plot_all(**outputs, x=plant.x_seq, x_ss=x_ss, a=plant.a_seq, a_ss=a_ss)


#%% Plot obsevations in 3D
if plant.y_seq.shape[0] > 1:
    raise ValueError('Too many trajectories')

def make_lims(a_seq, a_ss):
    hmin = min(a_ss[:,1].min(), plant.a_seq[:,:,1].min())
    hmax = max(a_ss[:,1].max(), plant.a_seq[:,:,1].max())

    rmin = min(a_ss[:,0].min(), plant.a_seq[:,:,0].min())
    rmax = max(a_ss[:,0].max(), plant.a_seq[:,:,0].max())

    hlim = (hmin, hmax)
    rlim = (rmin, rmax)
    return {'hlim': hlim, 'rlim': rlim}

fig, ax = f.plot_manifold(**make_lims(plant.a_seq, a_ss), r_samples=200, h_samples=50)
# fig, ax = f.plot_manifold()

_, ax = plot_parametric(plant.y_seq, mode='line', size=8, varname='y', ax=ax, cbar=False)#, t_on=t_on, t_off=t_off)
ax.scatter(*outputs['y'][:,0,:].T, marker='.', s=400, c='k')
# ax.text(*outputs['y'][:,0,:].squeeze()+0.1, '$y_0$')
ax.scatter(*y_ss.T, marker='*', s=400, c='k')
# ax.text(*y_ss.squeeze()+0.1, '$y^*$')
# ax.axis('off')

# axs[0,0].scatter(*x_ss.T, marker='*', s=250, c='k')
# axs[0,0].scatter(*x0.T, marker='.', s=250, c='k')
axs.scatter(*x_ss.T, marker='*', s=250, c='k')
axs.scatter(*x0.T, marker='.', s=250, c='k')

# fig.savefig('./results/grant/swiss.pdf', bbox_inches='tight', pad_inches=0, transparent=True)

#%% #%% Plot controlled outputs
fig, ax = plot_vs_time(plant.y_seq, target=y_ss, t_on=t_on, t_off=t_off, varname='y', mode='overlaid', legend=False)
ax[0].set_title('Swiss roll')
# ax[0].legend(ax[0].get_lines()[::2], ['$y_1(t)$','$y_2(t)$','$y_3(t)$'])
fig.set_size_inches(4,5)
fig.tight_layout()


#%%
fig, ax = f.plot_manifold(hlim=(plant.a_seq[:,:,1].min(), plant.a_seq[:,:,1].max()),
                          rlim=(plant.a_seq[:,:,0].min(), plant.a_seq[:,:,0].max()))
# _, ax = plot_parametric(plant.y_seq, mode='line', varname='y', ax=ax)#, t_on=t_on, t_off=t_off)

#%% Plot error heatmaps
#compute errors
err = {#'x': ( x_ss - plant.x_seq[:,-10:,:].mean(dim=1) ).norm(dim=1).reshape(len(x_ss_0), len(x_ss_1)),
       #'a': ( a_ss - plant.a_seq[:,-10:,:].mean(dim=1) ).norm(dim=1).reshape(len(x_ss_0), len(x_ss_1)),
       'y': ( y_ss - plant.y_seq[:,-2:-1,:].mean(dim=1) ).norm(dim=1).reshape(len(x_ss_0), len(x_ss_1))}
var_names = err.keys()

for varname in var_names:
    # v = eval(varname)
    # variance = (v - v.mean(dim=(0,1))).norm(dim=2).mean()
    # err[varname] = err[varname] / variance
    dist_to_go = (outputs['y'][:,0,:] - y_ss).norm(dim=1).reshape(len(x_ss_0), len(x_ss_1))
    err[varname] = err[varname] / dist_to_go

#fix cmap
cmap = matplotlib.colormaps['Reds']
cmap.set_over('grey')
vmax = defaultdict(lambda: None)
vmax.update({'x': 17, 'a': 17, 'y': None})

#plot
fig, axs = plt.subplots(1,len(var_names), sharex=True, sharey=True, figsize=(4*len(var_names),3.5), squeeze=False)
for varname,ax in zip(var_names, axs[0]):
    pcm = ax.pcolormesh(x_ss_0, x_ss_1, err[varname].T, vmin=0, vmax=vmax[varname], cmap=cmap)
    # ax.scatter(*x0[0], c='k')
    # ax.text(*x0[0]+0.15, '$x^{init}$')
    ax.axis('square')
    plt.colorbar(pcm, ax=ax, extend='max' if vmax[varname] is not None else None)
    ax.set_title(f'NMSE ${varname}(t)$')
    ax.set_xlabel('Target $x^{*}_0$')
    ax.set_ylabel('Target $x^{*}_1$')
# fig.suptitle(f"$Q={'' if Q==1 else f'{Q}\\times '}C^TC; R={'' if R==1 else f'{R}\\times '}I$")
fig.tight_layout()



# fig.savefig('./results/grant/swiss_mse.pdf', bbox_inches='tight', pad_inches=0, transparent=True)

#%% Histogram to eyeball good values for vmax in heatmaps
fig, axs = plt.subplots(1, len(var_names), sharex=True, figsize=(13,4))
for varname,ax in zip(var_names, axs):
    ax.hist(err[varname].flatten(), bins=100)
    ax.set_title(f'{varname} errors distribution')
fig.tight_layout()
