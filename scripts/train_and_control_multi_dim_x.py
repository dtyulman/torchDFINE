# %%
from copy import deepcopy
import argparse
from collections import defaultdict
import os
import pickle
os.environ['TQDM_DISABLE'] = '1'

import numpy as np
import matplotlib.pyplot as plt
import torch
torch.manual_seed(10)
from torch.utils.data import DataLoader
from controller import LQGController
from plot_utils import plot_parametric

from data.SSM import NonlinearStateSpaceModel, AffineTransformation, SwissRoll
from datasets import DFINEDataset
from trainers.TrainerDFINE import TrainerDFINE
from config_dfine import get_default_config
from time_series_utils import z_score_tensor, mse_from_encoding_dict


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
parser.add_argument('--no_input', required=False, default=False, action='store_true')
parser.add_argument('--dim_x', type=int, default=2)
parser.add_argument('--scale_forward_pred', type=float, default=0)
parser.add_argument('--manual_seed', type=int, default=10)
parser.add_argument('--gpu_id', type=str, default='2', help='set gpu id')
args = parser.parse_args()
# torch.manual_seed(args.manual_seed)
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_id
# %%
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

# Q = 1e-45 * torch.diag(torch.ones(dim_x))  #cf W in DFINE
# R = 1e-45 * torch.diag(torch.ones(dim_a))  #cf R in DFINE
# S = 1e-45 * torch.diag(torch.ones(dim_y))  
Q = 1e-2 * torch.diag(torch.ones(dim_x))  #cf W in DFINE
R = 1e-2 * torch.diag(torch.ones(dim_a))  #cf R in DFINE
S = 2e-3 * torch.diag(torch.ones(dim_y))  

ssm = NonlinearStateSpaceModel(A_fn,B,C,f,Q,R,S)

print(ssm)

# %%
use_saved_sim_data = True
num_seqs = 1024
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

simulation_save_path = 'simulations/swissroll' + (f"_u={'_'.join([str(train_input[key]) for key in train_input.keys()])}") + '_fixed'

if not use_saved_sim_data:
    xmin = -10
    xmax = 10
    x0 = torch.rand(num_seqs, dim_x, dtype=torch.float32)*(xmax-xmin)+xmin
    x,a,y = ssm.generate_trajectory(x0=x0, u_seq=u, num_seqs=num_seqs)
    simulation_dict = dict(zip(['x','a','y','u'],[x,a,y,u]))
else:
    # simulation_dict = pickle.load(open(simulation_save_path,'rb'))
    simulation_dict = pickle.load(open('/home/ebilgin/CalcDFINE/sim_dict.pkl','rb'))
    x = simulation_dict['x']
    a = simulation_dict['a']
    y = simulation_dict['y']
    u = simulation_dict['u']
# y, y_mean, y_std = z_score_tensor(y)

# %%
# fig, ax = f.plot_manifold(hlim=(a[:,:,1].min(), a[:,:,1].max()), rlim=(a[:,:,0].min(), a[:,:,0].max()))

ax_x = ax_a = ax_y = None
cb = True
for i in range(10):
    _, ax_x = plot_parametric(x[i], mode='line', ax=ax_x, add_cbar=cb, varname='x')
    _, ax_a = plot_parametric(a[i], mode='line', ax=ax_a, add_cbar=cb, varname='a')
    _, ax_y = plot_parametric(y[i], mode='line', ax=ax_y, add_cbar=cb, varname='y')
    cb = False

# fig, ax = plot_x(u[i])
# ax.set_ylabel('u')

# %%
use_ground_truth = False
load_model = False
# load_model = r"C:\Users\bilgi\Desktop\Research\torchDFINE\results\train_logs\2024-04-22\143825_u=binary_noise_-0.5_0.5_no_manifold_dim_x_5"
# load_model = r"C:\Users\bilgi\Desktop\Research\torchDFINE\results\train_logs\2024-04-19\191223_u=binary_noise_-0.5_0.5_no_manifold_dim_x_2"
# load_model = r"C:\Users\bilgi\Desktop\Research\torchDFINE\results\train_logs\2024-04-22\143825_u=binary_noise_-0.5_0.5_no_manifold_dim_x_5"
# load_model = r"C:\Users\bilgi\Desktop\Research\torchDFINE\results\train_logs\2024-04-22\142852_u=binary_noise_-0.5_0.5_dim_x_5"
# load_model = r"C:\Users\bilgi\Desktop\Research\torchDFINE\results\train_logs\2024-04-09\122258_u=binary_noise_-0.5_0.5"
# load_model = r"C:\Users\bilgi\Desktop\Research\torchDFINE\results\train_logs\2024-04-19\153421_u=binary_noise_-0.5_0.5_no_manifold_dim_x_5"

config = get_default_config()
config.model.no_manifold = args.no_manifold
config.model.fit_D_matrix = args.fit_D_matrix
config.model.dim_x = args.dim_x
config.model.dim_a = dim_y if config.model.no_manifold else dim_a
config.model.dim_u = dim_u
config.model.dim_y = dim_y
config.loss.scale_forward_pred = args.scale_forward_pred

no_manifold_key = '_no_manifold' if config.model.no_manifold else ''
feedthrough_key = '_feedthrough' if config.model.fit_D_matrix else ''
no_inp_key = '_no_inp' if args.no_input else ''

scale_forward_key = f'_scale_{args.scale_forward_pred}' if args.scale_forward_pred > 0 else ''
config.model.save_dir = config.model.save_dir + (f"_u={'_'.join([str(train_input[key]) for key in train_input.keys()])}") + no_manifold_key + feedthrough_key + scale_forward_key + no_inp_key
if args.no_input: u = torch.zeros_like(u)
config.model.save_dir = config.model.save_dir + '_dim_x_{}'.format(config.model.dim_x)
config.model.activation = None if config.model.no_manifold else 'tanh'
config.model.hidden_layer_list = None if config.model.no_manifold else [20,20,20,20]

config.train.num_epochs = 200
config.train.batch_size = 256
config.lr.scheduler = 'cyclic'
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
    
    encoding_dict = trainer.save_encoding_results(train_loader=train_loader,do_full_inference=False,save_results=False)
    mse_dict = mse_from_encoding_dict(encoding_dict=encoding_dict,steps_ahead=config.loss.steps_ahead)
    pickle.dump(mse_dict,open(config.model.save_dir + '/mse_dict.pkl'))    


# # %%
# import numpy as np
# np.set_printoptions(suppress=True)
# with torch.no_grad():
#     print('train_input:', train_input)
#     print('---')
#     print('A =', trainer.dfine.ldm.A.numpy())
#     print('B =', trainer.dfine.ldm.B.numpy())
#     print('W =', trainer.dfine.ldm._get_covariance_matrices()[0].numpy())
#     print('---')
#     print('C =', trainer.dfine.ldm.C.numpy())
#     print('R =', trainer.dfine.ldm._get_covariance_matrices()[1].numpy())
#     print()

# %%
# rlim=(0, 4*torch.pi)#(a[:,:,0].min(), a[:,:,0].max())
# hlim=(-1,1)#(a[:,:,1].min(), a[:,:,1].max())
# r = torch.linspace(rlim[0], rlim[1], 400)
# h = torch.linspace(hlim[0], hlim[1], 10)
# a_inputs_gt = torch.cartesian_prod(r, h)
# y_samples_gt = ssm.compute_observation(a_inputs_gt)

# a_inputs = a_inputs_gt
# with torch.no_grad():
#     y_samples = trainer.dfine.decoder(a_inputs)

# fig = plt.figure(figsize=(7,5))
# ax = fig.add_subplot(1,1,1, projection='3d' if dim_y==3 else None)
# im1 = ax.scatter(*y_samples_gt.T, c=a_inputs_gt[:,1], marker='.', cmap='coolwarm')
# im2 = ax.scatter(*y_samples.T,    c=a_inputs[:,1], marker='.', cmap='viridis')

# ax.set_title('Decoder outputs')
# ax.set_xlabel('$y_1$')
# if dim_y>1: ax.set_ylabel('$y_2$')
# if dim_y>2: ax.set_zlabel('$y_3$')
# fig.colorbar(im1, label='SSM manifold h input $d_h(t)$')
# fig.colorbar(im2, label='DFINE decoder h input $a_1(t)$')

# fig.tight_layout()

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

# %%
#     var_names = ['x', 'a', 'y']

#     model = trainer.dfine

#     plant = deepcopy(ssm)
#     plant.global_noise_toggle = False

#     controller = LQGController(plant, model)
#     num_steps = 50
#     Q = 1

#     for R in [.1,1,5]:
#         # Set start and target points. Must be set in plant's latent space `a` or `x` to ensure valid `y`
#         x0 = torch.tensor([0., 0.], dtype=torch.float32)
#         # x_ss_1 = torch.tensor([1.,])
#         # x_ss_2 = torch.tensor([1.,])
#         x_ss_1 = torch.linspace(-7, 7, 20)
#         x_ss_2 = torch.linspace(-7, 7, 20)
#         x_ss_list = torch.cartesian_prod(x_ss_1, x_ss_2)
#         idx_list = torch.cartesian_prod(torch.arange(len(x_ss_1)), torch.arange(len(x_ss_2)))

#         err = defaultdict(lambda: torch.full((len(x_ss_1), len(x_ss_2)), torch.nan))
#         for x_ss, (i,j) in zip(x_ss_list, idx_list):
#             # print(f'x_ss={x_ss.numpy()}, idx=({i},{j})')
#             _, a_ss, y_ss = controller.generate_observation(x_ss)
#             outputs = controller.run_control(y_ss, x0=x0, num_steps=num_steps, Q=Q, R=R)
#             outputs['x_ss'] = x_ss
#             outputs['a_ss'] = a_ss

#             err['x'][i,j] = (x_ss - outputs['x'][:,-5:,:].mean(dim=1)).norm()
#             err['a'][i,j] = (a_ss - outputs['a'][:,-5:,:].mean(dim=1)).norm()
#             err['y'][i,j] = (y_ss - outputs['y'][:,-5:,:].mean(dim=1)).norm()
    
#         results_dict[dim_x_sweep][f'control_mse_R_{R}'] = err['y'].mean()
#         pickle.dump(dict(err),open(config.model.save_dir + '\err_R_{}.pkl'.format(R),'wb')) # save error matrix



#         # fig, ax = controller.plot_all(**outputs)


#         # Barplot to eyeball good values for vmax in heatmaps
#         # fig, axs = plt.subplots(1, len(var_names), figsize=(13,4))
#         # for var,ax in zip(var_names, axs):
#         #     ax.hist(err[var].flatten())
#         #     ax.set_title(f'{var} errors distribution')
#         # fig.tight_layout()

#         cmap = plt.cm.get_cmap('Reds')
#         cmap.set_over('black')
#         vmax = defaultdict(lambda: None)
#         vmax.update({'x': 10, 'a': 10, 'y': 10})

#         fig, axs = plt.subplots(1,len(var_names), sharex=True, sharey=True, figsize=(13,4))
#         for var,ax in zip(var_names, axs):
#             pcm = ax.pcolormesh(x_ss_1, x_ss_2, err[var].T, vmin=0, vmax=vmax[var], cmap=cmap)
#             ax.scatter(*x0, c='k')
#             ax.text(*x0+0.15, '$x^{init}$')
#             ax.axis('square')
#             plt.colorbar(pcm, ax=ax, extend='max' if vmax[var] is not None else None)
#             ax.set_title(f'${var}$ error')
#             ax.set_xlabel('Target $x^{ss}_0$')
#             ax.set_ylabel('Target $x^{ss}_1$')
#         # fig.suptitle(f"$Q={'' if Q==1 else f'{Q}\\times '}C^TC; R={'' if R==1 else f'{R}\\times '}I$")
#         Q_str = '' if Q==1 else f'{Q}\\times '
#         R_str = '' if R==1 else f'{R}\\times '
#         fig.suptitle(f"$Q={Q_str}C^TC; R={R_str}I$")
#         fig.tight_layout()
#         fig.tight_layout()

#         fig.savefig(config.model.save_dir + '\R_{}.png'.format(R))
        
# results_path = 'results_dict' + no_manifold_key + feedthrough_key + scale_forward_key + '.pkl'
# pickle.dump(results_dict,open(results_path,'wb'))


    