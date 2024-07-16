# %%
from copy import deepcopy
import argparse
from collections import defaultdict
import os, sys
repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(repo_dir)
import pickle
os.environ['TQDM_DISABLE'] = '1'

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch

from torch.utils.data import DataLoader
from controller import LQGController
from plot_utils import plot_parametric

from data.SSM import NonlinearStateSpaceModel, SwissRoll
from datasets import DFINEDataset
from trainers.TrainerDFINE import TrainerDFINE
from config_dfine import get_default_config
from time_series_utils import z_score_tensor, mse_from_encoding_dict
from python_utils import Timer
from torch import nn
from modules.Autoencoder import Autoencoder

class WrapperModule(torch.nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f
        
    def forward(self, *args, **kwargs):
        return self.f(*args, **kwargs)
    
parser = argparse.ArgumentParser(description='MDFINE on Sabes Dataset')

# Simulation related settings
parser.add_argument('--no_manifold', required=False, default=False, action='store_true')
parser.add_argument('--fit_D_matrix', required=False, default=False, action='store_true')
parser.add_argument('--train_ae_seperately', required=False, default=True, action='store_true')
parser.add_argument('--train_on_manifold_latent', required=False, default=True, action='store_true')
parser.add_argument('--dim_x', type=int, default=2)
parser.add_argument('--scale_forward_pred', type=float, default=0)
parser.add_argument('--manual_seed', type=int, default=20)
parser.add_argument('--num_seqs', type=int, default=2**16)
args = parser.parse_args()
torch.manual_seed(args.manual_seed)
# %%
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

# Q = 1e-45 * torch.diag(torch.ones(dim_x))  #cf W in DFINE
# R = 1e-45 * torch.diag(torch.ones(dim_a))  #cf R in DFINE
# S = 1e-45 * torch.diag(torch.ones(dim_y))  
Q = 1e-2 * torch.diag(torch.ones(dim_x))  #cf W in DFINE
R = 1e-2 * torch.diag(torch.ones(dim_a))  #cf R in DFINE
S = 2e-3 * torch.diag(torch.ones(dim_y))  

ssm = NonlinearStateSpaceModel(A,b,B,C,f, Q,R,S)

print(ssm)

# %%
num_seqs = args.num_seqs
num_steps = 200

# train_input = {'type': 'none'}
# train_input = {'type': 'const', 'val': 0.2}
# train_input = {'type': 'gaussian', 'std': 2}
train_input = {'type': 'binary_noise', 'lo': -.5, 'hi': .5}
# train_input = {'type': 'multi_level_noise', 'lo':-3,'hi':3,'step':0.1, 'probabilities':None, 'equal_prob':True, 'each_input_duration': 1}

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
elif train_input['type'] == 'multi_level_noise':
    each_input_duration = train_input['each_input_duration']
    levels = torch.arange(train_input['lo'],train_input['hi']+train_input['step'],train_input['step'])
    num_levels = len(levels)
    level_dict = dict(enumerate(levels))
    if train_input['probabilities'] is not None:
        probabilities = torch.tensor(train_input['probabilities'])
    elif train_input['equal_prob']:
        probabilities = torch.ones(num_levels) * (1/num_levels)
    else:
        raise ValueError('Choose equal_prob option or provide probabilities')
    u_dist = torch.distributions.categorical.Categorical(probabilities)
    u_indices = u_dist.sample((num_seqs, int(num_steps/each_input_duration), dim_u))
    u_indices = torch.repeat_interleave(u_indices,repeats=each_input_duration,dim=1)
    u = []
    for x in u_indices.flatten():
        u.append(level_dict[x.item()])
    u = torch.tensor(u,dtype=torch.float32).reshape(num_seqs,num_steps,dim_u) # Proably there is a better way of assigning dictionary values
else:
    raise ValueError(f'Invalid train_input type: {train_input["type"]}')

xmin = -10
xmax = 10
x0 = torch.rand(num_seqs, dim_x, dtype=torch.float32)*(xmax-xmin)+xmin
x,a,y = ssm(x0=x0, u_seq=u, num_seqs=num_seqs)
simulation_dict = dict(zip(['x','a','y','u'],[x,a,y,u]))
# y, y_mean, y_std = z_score_tensor(y)


# %%
load_model = False
config = get_default_config()
config.model.no_manifold = args.no_manifold
config.model.fit_D_matrix = args.fit_D_matrix
config.loss.scale_forward_pred = args.scale_forward_pred
config.model.dim_x = args.dim_x
config.model.dim_a = dim_y if config.model.no_manifold else dim_a
config.model.dim_y = dim_y
config.model.dim_u = dim_u
config.train.num_epochs = 500
config.train.batch_size = 256
config.lr.scheduler = 'cyclic'
config.loss.scale_l2 = 0
config.model.activation = None if config.model.no_manifold else 'tanh'
config.model.hidden_layer_list = None if config.model.no_manifold else [20,20,20,20]

save_dir_keys = {}
save_dir_keys['no_manifold_key'] = '_no_manifold' if config.model.no_manifold else ''
save_dir_keys['feedthrough_key'] = '_feedthrough' if config.model.fit_D_matrix else ''
save_dir_keys['scale_forward_key'] = f'_scale_{args.scale_forward_pred}' if config.loss.scale_forward_pred > 0 else ''
save_dir_keys['dim_x_key'] = f'_dim_x_{config.model.dim_x}'
save_dir_keys['train_ae_seperately_key'] = '_train_ae_seperately' if args.train_ae_seperately else ''
save_dir_keys['train_on_manifold_latent_key'] = '_train_manifold' if args.train_on_manifold_latent else ''

config.model.save_dir +=(f"_u={'_'.join([str(train_input[key]) for key in train_input.keys()])}") # input key
config.model.save_dir += ''.join([x for x in save_dir_keys.values()]) # model parameters

if args.train_ae_seperately:
    # Train Autoencoder
    loss_arr = []
    batch_size_ae = 32
    train_data = DFINEDataset(y, u)
    train_loader = DataLoader(train_data, batch_size=batch_size_ae, shuffle=True)
    n_epochs = 500
    ae_kwargs = {'dim_y': dim_y, 'dim_a': dim_a, 'layer_list':[20,20,20,20],
                'activation_str':'tanh','nn_kernel_initializer':'xavier_normal'}
    autoencoder = Autoencoder(**ae_kwargs)
    loss_fn = nn.MSELoss()
    opt = torch.optim.Adam(autoencoder.parameters())
    with Timer():
        for i in range(n_epochs):
            running_loss = 0.
            for batch in train_loader:
                y_batch, _, _, _ = batch
                y_hat = autoencoder(y_batch)
                loss = loss_fn(y_batch, y_hat)
                opt.zero_grad()
                loss.backward()
                opt.step()
                running_loss += loss.item()

            loss_arr.append(running_loss/len(train_loader))

            if i%10 == 0:
                print(f'iter={i}, loss={loss:.4f}')

    if args.train_on_manifold_latent:
        config.model.dim_y = ae_kwargs['dim_a']
        config.model.hidden_layer_list = None
        config.model.no_manifold = True
        config.model.activation = None
        trainer_only_LDM = TrainerDFINE(config)

        a = autoencoder.encoder(y).detach()
        train_data = DFINEDataset(a, u)
        train_loader = DataLoader(train_data, batch_size=config.train.batch_size)
        trainer_only_LDM.train(train_loader)

        # Create a new DFINE object, load LDM from trained DFINE and encoder/decoder from previously trained autoencoder (after setting the necessary parameters)
        config.model.dim_y = ae_kwargs['dim_y']
        config.model.hidden_layer_list = ae_kwargs['layer_list']
        config.model.no_manifold = False
        config.model.activation = ae_kwargs['activation_str']
        train_data = DFINEDataset(y, u)
        train_loader = DataLoader(train_data, batch_size=config.train.batch_size)
        trainer = TrainerDFINE(config)

        trainer.dfine.encoder = autoencoder.encoder
        trainer.dfine.decoder = autoencoder.decoder
        trainer.dfine.ldm = trainer_only_LDM.dfine.ldm

        # last checkpoint in the save directory also includes encoder and decoder
        trainer._save_ckpt(epoch=config.train.num_epochs,
                            model=trainer.dfine,
                            optimizer=trainer.optimizer,
                            lr_scheduler=trainer.lr_scheduler)
    else:
        config.model.dim_y = ae_kwargs['dim_y']
        config.model.hidden_layer_list = ae_kwargs['layer_list']
        trainer = TrainerDFINE(config)

        # load trained autoencoder weights as DFINE encoder and decoder
        trainer.dfine.encoder = autoencoder.encoder
        trainer.dfine.decoder = autoencoder.decoder

        # freeze encoder/decoder weights
        for n,p in trainer.dfine.named_parameters():
            if (('encoder' in n) or ('decoder' in n)):
                p.requires_grad = False
            
        train_data = DFINEDataset(y, u)
        train_loader = DataLoader(train_data, batch_size=config.train.batch_size)
        trainer.train(train_loader)
    
else:
    trainer = TrainerDFINE(config)
    train_data = DFINEDataset(y, u)
    train_loader = DataLoader(train_data, batch_size=config.train.batch_size)
    trainer.train(train_loader)

# save final mse values
encoding_dict = trainer.save_encoding_results(train_loader=train_loader,do_full_inference=False,save_results=False)
mse_dict = mse_from_encoding_dict(encoding_dict=encoding_dict,steps_ahead=config.loss.steps_ahead)
torch.save(mse_dict,open(config.model.save_dir + '/mse_dict.pt'))    

#%%
model = deepcopy(trainer.dfine).requires_grad_(False)
plant = deepcopy(ssm)
# plant.Q_distr = plant.R_distr = plant.S_distr = None

controller = LQGController(plant, model)
num_steps = 100
t_on = 20#-1
t_off = 80#float('inf')
Q = 1

# Set start and target points. Must be set in plant's latent space `a` or `x` to ensure valid `y`
# x_ss_0 = torch.tensor([0.,])
# x_ss_1 = torch.tensor([0.,])
x_ss_0 = torch.linspace(-10, 10, 50)
x_ss_1 = torch.linspace(-10, 10, 50)

x_ss = torch.cartesian_prod(x_ss_0, x_ss_1)
a_ss = plant.compute_manifold_latent(x_ss)
y_ss = plant.compute_observation(a_ss)

x0 = torch.tensor([0., 0.], dtype=torch.float32).expand(x_ss.shape)

err_dict_path = os.path.join(config.model.save_dir,'error_dicts')
os.makedirs(err_dict_path,exist_ok=True)
for R in np.arange(.1,30,.1):
    outputs = controller.run_control(y_ss, {'x0':x0}, num_steps=num_steps, Q=Q, R=R, t_on=t_on, t_off=t_off)

    #% Plot controlled dynamics
    fig, ax = controller.plot_all(**outputs, x=plant.x_seq, x_ss=x_ss, a=plant.a_seq, a_ss=a_ss)

    # Plot error heatmaps
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
    
    Q_str = '' if Q==1 else f'{Q}\\times '
    fig.suptitle(f"$Q={Q_str}C^TC; R={'' if R==1 else f'{R}'}\\times I$")
    fig.tight_layout()
    fig.tight_layout()

    torch.save(dict(err),os.path.join(err_dict_path,f'R_{R}.pt'))