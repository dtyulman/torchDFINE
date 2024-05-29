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

A = torch.tensor([[.95,   0.05],
                  [.05,   .9]])
b = torch.tensor([0., 0])
B = torch.eye(dim_x, dim_u)
C = torch.eye(dim_a, dim_x)
f = SwissRoll(dim_y)

Q = 1e-2 * torch.diag(torch.ones(dim_x))  #cf W in DFINE
R = 1e-2 * torch.diag(torch.ones(dim_a))  #cf R in DFINE
S = 2e-3 * torch.diag(torch.ones(dim_y))

ssm = NonlinearStateSpaceModel(A,b,B,C,f, Q,R,S)

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
x0 = torch.rand(num_seqs, dim_x, dtype=torch.float32)*(xmax-xmin) + xmin
x,a,y = ssm(x0=x0, u_seq=u, num_seqs=num_seqs)
# y, y_mean, y_std = z_score_tensor(y)

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
load_model = "/Users/dtyulman/Drive/dfine_ctrl/torchDFINE/results/train_logs/2024-03-15/154837_u={'type': 'binary_noise', 'lo': -0.5, 'hi': 0.5}"
# load_model = '/Users/dtyulman/Drive/dfine_ctrl/torchDFINE/results/train_logs/2024-04-11/164847_u=binary_noise_-0.5_0.5'

config = get_default_config()
config.model.dim_x = dim_x
config.model.dim_u = dim_u
config.model.dim_a = dim_a
config.model.dim_y = dim_y
config.model.save_dir = config.model.save_dir + (f"_u={'_'.join([str(v) for v in train_input.values()])}")
config.model.hidden_layer_list = [20,20,20,20]

config.train.num_epochs = 200
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
    print('---')
    print('A =', trainer.dfine.ldm.A.numpy())
    print('B =', trainer.dfine.ldm.B.numpy())
    print('W =', trainer.dfine.ldm._get_covariance_matrices()[0].numpy())
    print('---')
    print('C =', trainer.dfine.ldm.C.numpy())
    print('R =', trainer.dfine.ldm._get_covariance_matrices()[1].numpy())
    print()


#%% Plot encoder/decoder outputs
#ground truth manifold
rlim=(0, 3*torch.pi)#(a[:,:,0].min(), a[:,:,0].max())
hlim=(-6,6)#(a[:,:,1].min(), a[:,:,1].max())
r = torch.linspace(rlim[0], rlim[1], 100)
h = torch.linspace(hlim[0], hlim[1], 10)
a_inputs_gt = torch.cartesian_prod(r, h)
y_samples_gt = ssm.compute_observation(a_inputs_gt, noise=False)

#learned manifold
rlim=(-8*torch.pi, 8*torch.pi)#(a[:,:,0].min(), a[:,:,0].max())
hlim=(-5,5)#(a[:,:,1].min(), a[:,:,1].max())
r = torch.linspace(rlim[0], rlim[1], 400)
h = torch.linspace(hlim[0], hlim[1], 10)
a_inputs = torch.cartesian_prod(r, h)
with torch.no_grad():
    y_samples = trainer.dfine.decoder(a_inputs)

#plotting
fig = plt.figure(figsize=(7,5))
ax = fig.add_subplot(1,1,1, projection='3d' if dim_y==3 else None)
im1 = ax.scatter(*y_samples_gt.T, marker='.', c='k')#a_inputs_gt[:,1], cmap='coolwarm')
im2 = ax.scatter(*y_samples.T,    marker='.', c='r')#a_inputs[:,1],    cmap='viridis')

ax.set_title('Decoder outputs')
ax.set_xlabel('$y_1$')
if dim_y>1: ax.set_ylabel('$y_2$')
if dim_y>2: ax.set_zlabel('$y_3$')
# fig.colorbar(im1, label='SSM manifold h input $d_h(t)$')
# fig.colorbar(im2, label='DFINE decoder h input $a_1(t)$')

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
model = deepcopy(trainer.dfine)
plant = deepcopy(ssm)
# plant.Q_distr = plant.R_distr = plant.S_distr = None

controller = LQGController(plant, model)
num_steps = 100
t_on = 20#-1
t_off = 80#float('inf')
Q = 1
R = .1

# Set start and target points. Must be set in plant's latent space `a` or `x` to ensure valid `y`
x_ss_0 = torch.tensor([2.,])
x_ss_1 = torch.tensor([2.,])
# x_ss_0 = torch.linspace(-10, 10, 30)
# x_ss_1 = torch.linspace(-10, 10, 30)

x_ss = torch.cartesian_prod(x_ss_0, x_ss_1)
a_ss = plant.compute_manifold_latent(x_ss)
y_ss = plant.compute_observation(a_ss)

x0 = torch.tensor([0., 0.], dtype=torch.float32).expand(x_ss.shape)
outputs = controller.run_control(y_ss, {'x0':x0}, num_steps=num_steps, Q=Q, R=R, t_on=t_on, t_off=t_off)

#% Plot controlled dynamics
fig, ax = controller.plot_all(**outputs, x=plant.x_seq, x_ss=x_ss, a=plant.a_seq, a_ss=a_ss)


#%% Plot error heatmaps
#compute errors
err = {'x': ( x_ss - plant.x_seq[:,-10:,:].mean(dim=1) ).norm(dim=1).reshape(len(x_ss_0), len(x_ss_1)),
       'a': ( a_ss - plant.a_seq[:,-10:,:].mean(dim=1) ).norm(dim=1).reshape(len(x_ss_0), len(x_ss_1)),
       'y': ( y_ss - plant.y_seq[:,-10:,:].mean(dim=1) ).norm(dim=1).reshape(len(x_ss_0), len(x_ss_1))}
var_names = err.keys()

#fix cmap
cmap = matplotlib.colormaps['Reds']
cmap.set_over('grey')
vmax = defaultdict(lambda: None)
vmax.update({'x': 17, 'a': 17, 'y': 10})

#plot
fig, axs = plt.subplots(1,len(var_names), sharex=True, sharey=True, figsize=(13,4))
for var,ax in zip(var_names, axs):
    pcm = ax.pcolormesh(x_ss_0, x_ss_1, err[var].T, vmin=0, vmax=vmax[var], cmap=cmap)
    ax.scatter(*x0[0], c='k')
    ax.text(*x0[0]+0.15, '$x^{init}$')
    ax.axis('square')
    plt.colorbar(pcm, ax=ax, extend='max' if vmax[var] is not None else None)
    ax.set_title(f'${var}$ error')
    ax.set_xlabel('Target $x^{ss}_0$')
    ax.set_ylabel('Target $x^{ss}_1$')
fig.suptitle(f"$Q={'' if Q==1 else f'{Q}\\times '}C^TC; R={'' if R==1 else f'{R}\\times '}I$")
fig.tight_layout()
fig.tight_layout()

#%% Histogram to eyeball good values for vmax in heatmaps
fig, axs = plt.subplots(1, len(var_names), sharex=True, figsize=(13,4))
for var,ax in zip(var_names, axs):
    ax.hist(err[var].flatten(), bins=100)
    ax.set_title(f'{var} errors distribution')
fig.tight_layout()
