from copy import deepcopy
from collections import defaultdict
import os
os.environ['TQDM_DISABLE'] = '1'

import torch
from torch.utils.data import DataLoader
import matplotlib as mpl
import matplotlib.pyplot as plt

from data.SSM import NonlinearStateSpaceModel, AffineTransformation, SwissRoll, plot_parametric
from datasets import ControlledDFINEDataset
from trainers.TrainerDFINE import TrainerDFINE
from config_dfine import get_default_config
from controller import LQGController


#%% Make a ground-truth SSM
dim_x = 2
dim_u = 2
dim_a = 2
dim_y = 3

A = torch.tensor([[.95,   0.05],
                  [.05,   .9]])
b = torch.tensor([0., 0])
A_fn = AffineTransformation(A,b)
B = torch.eye(dim_x, dim_u)
C = torch.eye(dim_a, dim_x)
f = SwissRoll(dim_y)

Q = 1e-2 * torch.diag(torch.ones(dim_x))  #cf W in DFINE
R = 1e-2 * torch.diag(torch.ones(dim_a))  #cf R in DFINE
S = 2e-3 * torch.diag(torch.ones(dim_y))

ssm = NonlinearStateSpaceModel(A_fn,B,C,f, Q,R,S)
ssm.global_noise_toggle = False
print(ssm)


#%% Train DFINE
load_model = False
# load_model = '/Users/dtyulman/Drive/dfine_ctrl/torchDFINE/results/train_logs/2024-04-11/170404_u=control'
# load_model = '/Users/dtyulman/Drive/dfine_ctrl/torchDFINE/results/train_logs/2024-04-11/173630_u=control'

config = get_default_config()
config.model.no_manifold = False #reduces to a linear model
config.model.dim_x = dim_x
config.model.dim_u = dim_u
config.model.dim_a = dim_y if config.model.no_manifold else dim_a
config.model.dim_y = dim_y
config.model.hidden_layer_list = None if config.model.no_manifold else [20,20,20,20]
config.model.activation = None if config.model.no_manifold else get_default_config().model.activation
config.model.save_dir = config.model.save_dir  + ('_linear' if config.model.no_manifold else ''
                                                  '_u=control'
                                                  f'_noise={ssm.global_noise_toggle}')

config.train.num_epochs = 200
config.train.batch_size = 32
config.lr.scheduler = 'constantlr'
config.loss.scale_l2 = 0

if load_model:
    config.model.save_dir = load_model
    config.load.ckpt = 100 #config.train.num_epochs

trainer = TrainerDFINE(config)
train_controller = LQGController(model=ssm, plant=ssm)
train_data = ControlledDFINEDataset(train_controller, R=5, umin=-5, umax=5, length=config.train.batch_size)

# y,u,b,m = next(train_data)

#%%
if not load_model:
    train_loader = DataLoader(train_data, batch_size=config.train.batch_size)
    trainer.train(train_loader)

#%% Plot sample batch
ax = defaultdict(lambda: None)
cb = True
train_data.dataset_len = 10
train_data.controller.plant.global_noise_toggle = False
for _ in train_data:
    for var in ['x', 'a', 'y', 'u']:
        _, ax[var] = plot_parametric(train_data.outputs[var][0], mode='line', ax=ax[var], add_cbar=cb if var=='y' else False, varname=var)
        ax[var].scatter(*train_data.outputs[f'{var}_ss'], marker='x', color='k')
    cb = False

#%% Plot sample trajectory
LQGController.plot_all(**train_data.outputs)


#%% Print model parameters
import numpy as np
np.set_printoptions(suppress=True)
with torch.no_grad():
    print('A =', trainer.dfine.ldm.A.numpy())
    print('B =', trainer.dfine.ldm.B.numpy())
    print('W =', trainer.dfine.ldm._get_covariance_matrices()[0].numpy())
    print('---')
    print('C =', trainer.dfine.ldm.C.numpy())
    print('R =', trainer.dfine.ldm._get_covariance_matrices()[1].numpy())
    print()


#%% Plot encoder/decoder outputs
rlim=(0, 4*torch.pi) #(a[:,:,0].min(), a[:,:,0].max())
hlim=(-1, 1) #(a[:,:,1].min(), a[:,:,1].max())
r = torch.linspace(rlim[0], rlim[1], 400)
h = torch.linspace(hlim[0], hlim[1], 10)
a_inputs_gt = torch.cartesian_prod(r, h)
y_samples_gt = ssm.compute_observation(a_inputs_gt)

a_inputs = a_inputs_gt
with torch.no_grad():
    y_samples = trainer.dfine.decoder(a_inputs)

fig = plt.figure(figsize=(7,5))
ax = fig.add_subplot(1,1,1, projection='3d' if dim_y==3 else None)
im1 = ax.scatter(*y_samples_gt.T, c=a_inputs_gt[:,1], marker='.', cmap='coolwarm')
im2 = ax.scatter(*y_samples.T,    c=a_inputs[:,1], marker='.', cmap='viridis')

ax.set_title('Decoder outputs')
ax.set_xlabel('$y_1$')
if dim_y>1: ax.set_ylabel('$y_2$')
if dim_y>2: ax.set_zlabel('$y_3$')
fig.colorbar(im1, label='SSM manifold h input $d_h(t)$')
fig.colorbar(im2, label='DFINE decoder h input $a_1(t)$')

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
var_names = ['x', 'a', 'y']

model = trainer.dfine
plant = deepcopy(ssm)
plant.global_noise_toggle = False

controller = LQGController(plant, model)
num_steps = 50
Q = 1
R = 20

# Set start and target points. Must be set in plant's latent space `a` or `x` to ensure valid `y`
x0 = torch.tensor([5., 5.], dtype=torch.float32)
# x_ss_1 = torch.tensor([5.,])
# x_ss_2 = torch.tensor([0.,])
x_ss_1 = torch.linspace(-10, 10, 20)
x_ss_2 = torch.linspace(-10, 10, 20)
x_ss_list = torch.cartesian_prod(x_ss_1, x_ss_2)
idx_list = torch.cartesian_prod(torch.arange(len(x_ss_1)), torch.arange(len(x_ss_2)))

err = defaultdict(lambda: torch.full((len(x_ss_1), len(x_ss_2)), torch.nan))
for x_ss, (i,j) in zip(x_ss_list, idx_list):
    print(f'x_ss={x_ss.numpy()}, idx=({i},{j})')
    _, a_ss, y_ss = controller.generate_observation(x_ss)
    outputs = controller.run_control(y_ss, x0=x0, num_steps=num_steps, Q=Q, R=R)
    outputs['x_ss'] = x_ss
    outputs['a_ss'] = a_ss

    err['x'][i,j] = (x_ss - outputs['x'][:,-5:,:].mean(dim=1)).norm()
    err['a'][i,j] = (a_ss - outputs['a'][:,-5:,:].mean(dim=1)).norm()
    err['y'][i,j] = (y_ss - outputs['y'][:,-5:,:].mean(dim=1)).norm()


if len(x_ss_1) == len(x_ss_2) == 1:
    # Plot controlled dynamics
    fig, ax = LQGController.plot_all(**outputs)


#%% Plot error heatmaps
# Barplot to eyeball good values for vmax in heatmaps
fig, axs = plt.subplots(1, len(var_names), figsize=(13,4))
for var,ax in zip(var_names, axs):
    ax.hist(err[var].flatten())
    ax.set_title(f'{var} errors distribution')
fig.tight_layout()

#%%
cmap = mpl.colormaps['Reds']
cmap.set_over('grey')
vmax = {'x': None, 'a': None, 'y': None} #set any entry to None to remove vmax limit

fig, axs = plt.subplots(1,len(var_names), sharex=True, sharey=True, figsize=(13,4))
for var,ax in zip(var_names, axs):
    pcm = ax.pcolor(x_ss_1, x_ss_2, err[var].T, vmin=0, vmax=vmax[var], cmap=cmap)
    ax.scatter(*x0, c='k')
    ax.text(*x0+0.15, '$x^{init}$')
    ax.axis('square')
    plt.colorbar(pcm, ax=ax, extend='max' if vmax[var] is not None else None)
    ax.set_title(f'${var}$ error')
    ax.set_xlabel('Target $x^{ss}_0$')
    ax.set_ylabel('Target $x^{ss}_1$')
fig.suptitle(f"$Q={'' if Q==1 else f'{Q}\\times '}C^TC; R={'' if R==1 else f'{R}\\times '}I$")
fig.tight_layout()
fig.tight_layout()
