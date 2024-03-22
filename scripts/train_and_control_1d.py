from copy import deepcopy
import os
os.environ['TQDM_DISABLE'] = '1'

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.distributions.multivariate_normal import MultivariateNormal
import control

from data.SSM import NonlinearStateSpaceModel, AffineTransformation, RingManifold, IdentityManifold, plot_x, plot_y
from DFINE import DFINE
from datasets import DFINEDataset
from trainers.TrainerDFINE import TrainerDFINE
from config_dfine import get_default_config
from time_series_utils import z_score_tensor


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
A_fn = AffineTransformation(A,b)
B = torch.eye(dim_x, dim_u)
C = torch.eye(dim_a, dim_x)
# f = IdentityManifold(dim_y)
f = RingManifold(dim_y)

# Q = 1e-45 * torch.diag(torch.ones(dim_x))  #cf W in DFINE
# R = 1e-45 * torch.diag(torch.ones(dim_a))  #cf R in DFINE
# S = 1e-45 * torch.diag(torch.ones(dim_y))  
Q = 1e-2 * torch.diag(torch.ones(dim_x))  #cf W in DFINE
R = 1e-2 * torch.diag(torch.ones(dim_a))  #cf R in DFINE
S = 2e-3 * torch.diag(torch.ones(dim_y))  

ssm = NonlinearStateSpaceModel(A_fn,B,C,f, Q,R,S)

print(ssm)
#%% Generate data
num_seqs = 1#024 
num_steps = 200

# train_input = {'type': 'none'}
train_input = {'type': 'const', 'val': 0.2}
# train_input = {'type': 'gaussian', 'std': 2}
# train_input = {'type': 'binary_nouse', 'lo': 0, 'hi': 0.2}

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
    
x0 = torch.rand(num_seqs, dim_x, dtype=torch.float32)-0.5
x,a,y = ssm.generate_trajectory(x0=x0, u_seq=u, num_seqs=num_seqs)
# y, y_mean, y_std = z_score_tensor(y)

#%% Plot data sample
i = 0 #select sequence number, (0 .. num_seqs-1)
fig, ax = f.plot_manifold()
plot_y(y[i], mode='line', ax=ax)
# plot_x(x[i])
# fig, ax = plot_x(u[i])
# ax.set_ylabel('u')


#%% Train DFINE
use_ground_truth = False
load_model = False #'/Users/dtyulman/Drive/dfine_ctrl/torchDFINE/results/train_logs/2024-03-13/130907_u=binary_noise'

config = get_default_config()
config.model.dim_x = dim_x
config.model.dim_u = dim_u
config.model.dim_a = dim_a
config.model.dim_y = dim_y
config.model.save_dir = config.model.save_dir + (f'_u={train_input}')
config.model.hidden_layer_list = [20,20,20,20]

config.train.num_epochs = 100 
config.train.batch_size = 32
config.lr.scheduler = 'constantlr'
config.loss.scale_l2 = 0

if load_model:
    config.model.save_dir = load_model
    config.load.ckpt = config.train.num_epochs

trainer = TrainerDFINE(config)

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
y_samples_ssm = ssm._compute_y(a_inputs_ssm)

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
model = trainer.dfine 
model.requires_grad_(False)

plant = ssm
# plant = deepcopy(model)

num_trials = 1
num_steps = 100
t_ctrl_start = 20
t_ctrl_stop = num_steps - 20

# Logging
x     = torch.full((num_trials, num_steps, plant.dim_x), torch.nan) #plant dynamic latent
a     = torch.full((num_trials, num_steps, plant.dim_a), torch.nan) #plant manifold latent (not used, just tracking for posterity)
x_hat = torch.full((num_trials, num_steps, model.dim_x), torch.nan) #Kalman filter estimate of dynamic latent (except at t=0, which is x_hat[0]=C_inv*f_enc(y))
a_hat = torch.full((num_trials, num_steps, model.dim_a), torch.nan) #model manifold latent
y     = torch.full((num_trials, num_steps, plant.dim_y), torch.nan) #observations 
u     = torch.full((num_trials, num_steps, model.dim_u), torch.nan) #control input 

# Set start and target points. Must be set in plant's manifold latent space `a` to ensure valid points `y` in observation space
a0   = torch.tensor([-3.], dtype=torch.float32)
a_ss = torch.tensor([3.], dtype=torch.float32)

if isinstance(plant, DFINE):
    y0   = plant.decoder(a0)
    y_ss = plant.decoder(a_ss)
elif isinstance(plant, NonlinearStateSpaceModel):
    y0   = plant._compute_y(a0,   noise=False)
    y_ss = plant._compute_y(a_ss, noise=False)
else:
    raise ValueError('Invalid plant')    

# Get start and target points in model's latent space
a_hat0   = model.encoder(y0.unsqueeze(0)) #approx model manifold latent from init/target observation 
a_hat_ss = model.encoder(y_ss.unsqueeze(0))

C = model.ldm.C[0:1, ...]
x_hat0 = torch.linalg.solve(C, a_hat0) #i.e. x_hat = Cinv @ a_hat (TODO: is C generally invertible?)
#TODO: how to guarantee that this point in latent space actually corresponds to the desired target y_ss? 
#And, if we can do this for (x_hat_ss, y_ss) why can't we do it for all (x_hat_t, y_t) 
x_hat_ss = torch.linalg.solve(C, a_hat_ss)

# Compute steady-state control input
A = model.ldm.A[0:1, ...]
B = model.ldm.B[0:1, ...]
I = torch.eye(model.dim_x)
u_ss = torch.linalg.pinv(B) @ (I-A) @ x_hat_ss     

# Compute infinite-horizon LQR feedback gain matrix G #TODO: should this be finite horizon?
# J = \sum_t x^T Q x + u^T R u  
Q_lqr = C.T @ C  #effectively puts the penalty on a_hat with Q=I because a=Cx
R_lqr = 1*torch.eye(model.dim_u)
_, _, G = control.dare(A, B, Q_lqr, R_lqr) #TODO: should be computed for entire sequence A_t, B_t
G = torch.tensor(G, dtype=torch.float32)    

# Run controller
x[:,0,:] = x_hat0 #assume plant's initial latent state equals model's initial state #TODO: why is this reasonable?
a[:,0,:] = float('nan') 
y[:,0,:] = float('nan')
x_hat[:,0,:] = x_hat0
a_hat[:,0,:] = a_hat0

for t in range(num_steps-1):
    # Compute LQR control input
    if t_ctrl_start <= t <= t_ctrl_stop:
        u[:,t,:] = -G @ (x_hat[:,t,:] - x_hat_ss) + u_ss #TODO check dimensions, why is HY transposing?
    else:
        u[:,t,:] = 0
    
    # Step dynamics of plant with control input
    if isinstance(plant, DFINE):
        A, B, C = plant.ldm.A[0:1, ...], plant.ldm.B[0:1, ...], plant.ldm.C[0:1, ...]
        W, R = plant.ldm._get_covariance_matrices()
        x[:,t+1,:] = A @ x[:,t,:] + B @ u[:,t,:] + MultivariateNormal(torch.zeros(dim_x), W).sample()
        a[:,t+1,:] = C @ x[:,t+1,:] + MultivariateNormal(torch.zeros(dim_a), R).sample()
        y[:,t+1,:] = plant.decoder(a[:,t+1,:])                
    elif isinstance(plant, NonlinearStateSpaceModel):
        x[:,t+1,:], a[:,t+1,:], y[:,t+1,:] = plant.step(x[:,t,:], u[:,t,:])
    else:
        raise ValueError('Invalid plant')
    
    # Get model manifold latent from observation 
    # y[:,t+1,:] = z_score_tensor(y[:,t+1,:], fit=False, mean=y_mean, std=y_std)[0]
    a_hat[:,t+1,:] = model.encoder(y[:,t+1,:]) 
    
    # Estimate dynamic latent state from manifold latent state by Kalman filter #TODO: do this recursively, try control.dlqe()
    x_hat[:,t+1,:] = model.ldm(a=a_hat[:, 1:t+2, :], u=u[:, :t+1, :])[1][:,-1,:] 


#%% Plot latents, observations, control vs time
fig, ax = plt.subplots(3,2, sharex=True)
ax[0,0].axhline(x_hat_ss, c='tab:blue',      ls='--', label='$\\widehat{x}_{ss}$ target (est)')
ax[0,0].plot(   x[0],     c='tab:orange',          label='$x(t)$ plant')
ax[0,0].plot(   x_hat[0], c='tab:orange', ls='--', label='$\\widehat{x}(t)$ estimate (KF)')
ax[0,0].set_ylabel('Dynamic latent $x(t)$')
ax[0,0].legend()

ax[1,0].axhline(a_ss,     c='tab:blue',               label='${a}_{ss}$ target (true)')
ax[1,0].plot(   a[0],     c='tab:orange',          label='$a(t)$ plant')
ax[1,0].axhline(a_hat_ss, c='tab:blue',      ls='--', label='$\\widehat{a}_{ss}$ target (est)')
ax[1,0].plot(   a_hat[0], c='tab:orange', ls='--', label='$\\widehat{a}(t)$ estimate ($f_{enc})$')
ax[1,0].set_ylabel('Manifold latent $a(t)$')
ax[1,0].legend()

ax[2,0].axhline(u_ss, c='tab:blue',      label='steady_state')
ax[2,0].plot(u[0],    c='tab:orange', label='control_input')
ax[2,0].set_ylabel('Control input $u(t)$')
ax[2,0].legend()

for ydim in range(min(dim_y, 3)):
    ax[ydim,1].axhline(y_ss[ydim],  c='tab:blue',      label='$y_{ss}$ target')
    ax[ydim,1].plot(   y[0,:,ydim], c='tab:orange', label='$y(t)$ plant')
    ax[ydim,1].set_ylabel(f'Observation $y_{ydim+1}(t)$')
ax[0,1].legend()

for axis in ax.flatten():
    axis.axvline(t_ctrl_start, color='k', ls='--')
    axis.axvline(t_ctrl_stop, color='k', ls='--')

for axis in ax[-1,:]:
    axis.set_xlabel('Time')

fig.set_size_inches(11,8)
fig.tight_layout()

#%% Plot obsevations in 3D
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
f.plot_manifold(ax=ax)
plot_y(y[0], t_on=t_ctrl_start, t_off=t_ctrl_stop, ax=ax, mode='line')
ax.scatter(*y0, s=100, c='k')
ax.text(*(y0+0.1), '$t_{on}$')
ax.scatter(*y_ss, marker='x', s=100, c='k')    
ax.text(*(y_ss+0.1), '$t_{off}$')








