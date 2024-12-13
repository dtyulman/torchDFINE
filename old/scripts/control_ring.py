from copy import deepcopy
from collections import defaultdict
import os
os.environ['TQDM_DISABLE'] = '1'

import matplotlib
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from data.SSM import NonlinearStateSpaceModel, SwissRoll, RingManifold
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
dim_x = 1
dim_u = 1
dim_a = 1
dim_y = 3

A = torch.tensor([[0.99]])
b = torch.tensor([0.])
B = torch.eye(dim_x, dim_u)
C = torch.eye(dim_a, dim_x)
# f = IdentityManifold(dim_y)
f = RingManifold(dim_y)

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

xmin = 0
xmax = 2*torch.pi
x0 = torch.rand(num_seqs, dim_x)*(xmax-xmin) + xmin
x,a,y = ssm(x0=x0, u_seq=u, num_seqs=num_seqs)

#%% Plot data sample
i = 0 #select sequence number, (0 .. num_seqs-1)
fig, ax = f.plot_manifold()
plot_parametric(y[i], mode='line', ax=ax, varname='y')


#%% Train DFINE
use_ground_truth = False
load_model = '/Users/dtyulman/Drive/dfine_ctrl/torchDFINE/results/train_logs/2024-03-13/130907_u=binary_noise'

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
with torch.no_grad():
    print('train_input =', train_input)
    print('---')
    print('A =', trainer.dfine.ldm.A.numpy())
    print('B =', trainer.dfine.ldm.B.numpy())
    print('W =', trainer.dfine.ldm._get_covariance_matrices()[0].numpy())
    print('---')
    print('C =', trainer.dfine.ldm.C.numpy())
    print('R =', trainer.dfine.ldm._get_covariance_matrices()[1].numpy())
    print()

#%% Plot encoder/decoder outputs
a_inputs = torch.linspace(-10, 10, 1000).unsqueeze(-1)
a_inputs_gt = torch.pi*torch.arange(-2,3).unsqueeze(-1)
a_inputs_ssm = torch.linspace(0, 2*torch.pi, 1000).unsqueeze(-1)

with torch.no_grad():
    y_samples = trainer.dfine.decoder(a_inputs)
    y_samples_gt = trainer.dfine.decoder(a_inputs_gt)
y_samples_ssm = ssm.compute_observation(a_inputs_ssm)

if dim_y == 1:
    y_samples = torch.concatenate([torch.ones_like(y_samples), y_samples], dim=-1)
    y_samples_gt = torch.concatenate([torch.ones_like(y_samples_gt), y_samples_gt], dim=-1)
    y_samples_ssm = torch.concatenate([torch.ones_like(y_samples_ssm)*1.1, y_samples_ssm], dim=-1)

fig = plt.figure(figsize=(12,5))
ax = fig.add_subplot(1,2,1, projection='3d' if dim_y==3 else None)
im1 = ax.scatter(*y_samples.T,     c=a_inputs,     marker='.', cmap='viridis')
im3 = ax.scatter(*y_samples_ssm.T, c=a_inputs_ssm, marker='.', cmap='coolwarm')
for aigt,ysgt in zip(a_inputs_gt, y_samples_gt):
    ax.scatter(*ysgt.T, marker='x', c='k', s=80)
    ax.text(*ysgt.T, f'  $a={float(aigt):.3g}$')

ax.set_title('Decoder outputs')
ax.set_xlabel('$y_1$')
if dim_y>1: ax.set_ylabel('$y_2$')
if dim_y>2: ax.set_zlabel('$y_3$')
fig.colorbar(im1, label='DFINE Decoder input $a(t)$')
fig.colorbar(im3, label='SSM Manifold input $a(t)$')

y_inputs = torch.rand(10000, dim_y)*100-50
with torch.no_grad():
    a_samples = trainer.dfine.encoder(y_inputs)

ax = fig.add_subplot(1,2,2)
ax.scatter(y_inputs.norm(dim=1), a_samples, marker='.')
ax.set_title('Encoder outputs')
ax.set_ylabel('$a(t)$')
ax.set_xlabel('$||y(t)||$')

fig.tight_layout()

#%% Control
model = deepcopy(trainer.dfine).requires_grad_(False)
plant = deepcopy(ssm)
plant.Q_distr = plant.R_distr = plant.S_distr = None

controller = LQGController(plant, model)
num_steps = 50
t_on = 0
t_off = 50
horizon = 50#float('inf')

Q = 0
R = 1
F = 10000

# Set start and target points. Must be set in plant's latent space `a` or `x` to ensure valid `y`
# x_ss = torch.linspace(0, 2*torch.pi*49/50, 50).unsqueeze(-1)
x_ss = torch.tensor([[3.]])
a_ss = plant.compute_manifold_latent(x_ss)
y_ss = plant.compute_observation(a_ss)

x0 = torch.tensor([0.,]).expand(x_ss.shape)
outputs = controller.run_control(y_ss, {'x0':x0}, num_steps=num_steps, F=F, Q=Q, R=R, t_on=t_on, t_off=t_off, horizon=horizon)

#% Plot controlled dynamics
fig, ax = controller.plot_all(**outputs, x=plant.x_seq, x_ss=x_ss, a=plant.a_seq, a_ss=a_ss)

#%% Plot obsevations in 3D
# fig, ax = f.plot_manifold(hlim=(plant.a_seq[:,:,1].min(), plant.a_seq[:,:,1].max()),
#                           rlim=(plant.a_seq[:,:,0].min(), plant.a_seq[:,:,0].max()))
fig, ax = f.plot_manifold(samples=1000)

_, ax = plot_parametric(plant.y_seq, mode='line', size=8, varname='y', ax=ax, cbar=False)#, t_on=t_on, t_off=t_off)
ax.scatter(*outputs['y'][:,0,:].T, marker='.', s=400, c='k')
# ax.text(*outputs['y'][:,0,:].squeeze()+0.1, '$y_0$')
ax.scatter(*y_ss.T, marker='*', s=400, c='k')
# ax.text(*y_ss.squeeze()+0.1, '$y^*$')
# ax.axis('off')

axs.scatter(x_ss, 0, marker='*', s=250, c='k')
axs.scatter(x0, 0, marker='.', s=250, c='k')


# fig.savefig('./results/grant/ring.pdf', bbox_inches='tight', pad_inches=0, transparent=True)


#%% #%% Plot controlled outputs
fig, ax = plot_vs_time(plant.y_seq, target=y_ss, t_on=t_on, t_off=t_off, varname='y', mode='overlaid', legend=False)
ax[0].set_title('Ring-like')
# ax[0].legend(ax[0].get_lines()[::2], ['$y_1(t)$','$y_2(t)$','$y_3(t)$'])
fig.set_size_inches(4,5)
fig.tight_layout()


#%%
y_mse = ( y_ss - plant.y_seq[:,-2:-1,:].mean(dim=1) ).norm(dim=1)
# y_var = (y - y.mean(dim=(0,1))).norm(dim=2).mean()
dist_to_go = (outputs['y'][:,0,:] - y_ss).norm(dim=1)
y_nmse = y_mse / dist_to_go
fig, axs = plt.subplots()
axs.plot(x_ss.squeeze().numpy(), y_nmse.numpy(), color='r')
axs.set_xlabel('Target $x^*$')
axs.set_ylabel('NMSE y(t)')
# ax.axvline(3, ls='--', c='k')

# fig.savefig('./results/grant/ring_mse.pdf', bbox_inches='tight', pad_inches=0, transparent=True)
