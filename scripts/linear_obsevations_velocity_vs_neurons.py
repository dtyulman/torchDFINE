from importlib import reload
from collections import defaultdict
from copy import deepcopy
from pprint import pprint
import os
os.environ['TQDM_DISABLE'] = '1'

import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import control

from plot_utils import plot_parametric, plot_vs_time, plot_eigvals
from data.SSM import NonlinearStateSpaceModel, IdentityManifold
import config_dfine
from datasets import DFINEDataset
from trainers.TrainerDFINE import TrainerDFINE
from controller import LQGController
from time_series_utils import generate_input_noise
from python_utils import Timer

torch.set_default_dtype(torch.float64)

class WrapperModule(torch.nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, *args, **kwargs):
        return self.f(*args, **kwargs)


def init_dfine_from_ssm(dfine, ssm):
    assert dfine.dim_x == ssm.dim_x
    assert dfine.dim_a == ssm.dim_a
    assert dfine.dim_y == ssm.dim_y
    assert dfine.dim_u == ssm.dim_u

    dfine.ldm.A.data = ssm.A
    dfine.ldm.B.data = ssm.B
    dfine.ldm.C.data = ssm.C

    dfine.ldm.W_log_diag.data = torch.diag(torch.log(ssm.Q_distr.covariance_matrix))
    dfine.ldm.R_log_diag.data = torch.diag(torch.log(ssm.R_distr.covariance_matrix))
    assert ssm.S_distr is None

    dfine.decoder = ssm.f
    dfine.encoder = ssm.f.inv

    return dfine


def bias(seq, target, k=None):
    """Average bias over the final k timesteps
    seq: [B,T,N], target: [B,N]"""
    k = seq.shape[1] if k is None else k
    bias = (seq[:,-k:,:].mean(dim=1) - target).norm(dim=1) #([B,N]-[B,N]) -> [B]
    return bias


def mse(seq, target, k=None):
    """MSE over the final k timesteps
    seq: [B,T,N], target: [B,N]"""
    k = seq.shape[1] if k is None else k
    mse = (seq[:,-k:,:] - target.unsqueeze(1)).norm(dim=(1,2)) #([B,k,N]-[B,1,N]) -> [B]
    return mse

#%% Instantiate LDM
use_as_observation = 'y'
assert use_as_observation in ['x', 'y']
print(f'Using observations: {use_as_observation}')

dim_x = 16
dim_u = dim_x
dim_a = 2 if use_as_observation=='y' else dim_x
dim_y = dim_a

force_stable_thres = None#0.999
rank_A = 16
assert rank_A <= dim_x

i=0
while True:
    if rank_A == dim_x:
        A = torch.rand(dim_x, dim_x)
    else:
        A = 0
        for i in range(rank_A):
            A += torch.outer(torch.rand(dim_x), torch.rand(dim_x))/rank_A
    if force_stable_thres is not None: #move eigenvalues of A above force_stable_thres to below it
        L,V = torch.linalg.eig(A)
        L[L.abs()>force_stable_thres] = L[L.abs()>force_stable_thres]/L[L.abs()>force_stable_thres].abs()
        A = V @ torch.diag(L) @ torch.linalg.inv(V)
        A = A.real

    b = None
    B = torch.eye(dim_x, dim_u)
    C = torch.eye(dim_a, dim_x)
    f = IdentityManifold(dim_y)
    Q = 1e-10 * torch.eye(dim_x)
    R = 1e-10 * torch.eye(dim_a)
    ssm = NonlinearStateSpaceModel(A,b,B,C,f,Q,R)

    if ssm.is_controllable(method='direct') and ssm.is_observable(method='direct') \
    and ssm.is_controllable(method='hautus') and ssm.is_observable(method='hautus'):
        break
    i += 1
print(i)


C_readout = torch.rand(2, dim_x) if use_as_observation=='x' else None

#%% Print/plot observability/controllability information

ssm.is_observable(verbose=True)
ssm.is_controllable(verbose=True)
ssm.is_output_controllable(verbose=True)
eigvals = torch.linalg.eigvals(A)
print(f'eig(A) = {eigvals[eigvals.abs()>1e-15].numpy()}')
print(f'rank(A) = {np.linalg.matrix_rank(A)}')
fig, ax = plot_eigvals(A, title=f'eig(A)')#'={eigvals[eigvals.abs()>1e-15].numpy()}')


obsv = control.obsv(A.numpy(), C.numpy());
_,So,_ = np.linalg.svd(obsv)
fig, ax = plt.subplots(2,1, sharex=True);
ax[0].stem(So)
ax[0].set_title(f'Observability matrix (rank={np.linalg.matrix_rank(obsv)})')


ctrb = control.ctrb(A.numpy(), B.numpy());
_,Sc,_ = np.linalg.svd(ctrb)
ax[1].stem(Sc)
ax[1].set_title(f'Controllability matrix (rank={np.linalg.matrix_rank(ctrb)})')
ax[1].set_xlabel('Singular values')

for a in ax:
    a.set_ylabel('Magnitude')
    a.set_yscale('log')
    a.minorticks_on()
    a.grid(visible=True, which='both', axis='y')

fig.tight_layout()
#%% Generate data
num_seqs = 2**16
num_steps = 100

u = generate_input_noise(dim_u, num_steps, num_seqs=num_seqs, lo=-0.5, hi=0.5, levels=2)
x,a,y = ssm(u_seq=u)


#%% Train DFINE using LDM's either x or y as observation
#model parameters

model_dim_x = 16 #dim_x

reload(config_dfine) #ensures timestamp in get_default_config().model.savedir is current
config = config_dfine.get_LDM_config(dim_x=model_dim_x, dim_a=dim_y, dim_u=dim_u)
config.model.save_dir = config.model.save_dir + '_ldm' + f'_{use_as_observation}'

#training parameters
config.train.num_epochs = 10
config.train.batch_size = 64

config.lr.scheduler = 'constantlr'
config.lr.init = 0.001 #default for Adam
config.loss.scale_l2 = 0
config.optim.grad_clip = float('inf')


#%%
use_ground_truth = True
load_model = False

if load_model:
    config.model.save_dir = load_model
    config.load.ckpt = config.train.num_epochs
    config.load.resume_train = True

trainer = TrainerDFINE(config)

if not load_model:
    if use_ground_truth:
        trainer.dfine = init_dfine_from_ssm(trainer.dfine, ssm)
    else:
        train_data = DFINEDataset(y=y, u=u)
        train_loader = DataLoader(train_data, batch_size=config.train.batch_size)
        with Timer():
            trainer.train(train_loader)

print('Nx =', model_dim_x, ', rank(WA) =', rank_A, ', obs =', use_as_observation)
#%% Control LDM
R_list = [1.,]#np.logspace(-10, 10, 21)
err_list = []
for R in R_list:
    model = deepcopy(trainer.dfine).requires_grad_(False)
    plant = deepcopy(ssm)
    # plant.Q_distr = plant.R_distr = plant.S_distr = None

    controller = LQGController(plant, model)
    num_steps = 50
    t_on = 0
    t_off = float('inf')
    Q = 1

    if len(R_list) == 1:
        y_ss = torch.tensor([[10., 10.]])
        shape = (1,1)

        # y_ss_0 = torch.linspace(-10, 10, 20)
        # y_ss_1 = torch.linspace(-10, 10, 20)
        # shape = (len(y_ss_0), len(y_ss_1))
        # y_ss = torch.cartesian_prod(y_ss_0, y_ss_1)
    else:
        # y_ss = torch.ones(1, 2)*10
        shape = (1,1)

    if use_as_observation == 'x':
        y_ss = (torch.linalg.pinv(C_readout) @ y_ss.unsqueeze(-1)).squeeze(-1)

    x0 = torch.tensor([10.]*2 + [0.]*14).unsqueeze(0)
    ctrlr = 'LQG' #lambda x,t: torch.randn(x.shape[0], dim_u)
    outputs = controller.run_control(y_ss, plant_init={'x0':x0}, num_steps=num_steps, R=R, t_on=t_on, t_off=t_off,
                                     ground_truth_latents='', penalize_a=True, controller=ctrlr)

    if use_as_observation == 'x':
        outputs['y_hat_ss'] = (C_readout @ outputs['y_hat_ss'].unsqueeze(-1)).squeeze(-1)
        outputs['y_ss'] = (C_readout @ outputs['y_ss'].unsqueeze(-1)).squeeze(-1)
        outputs['y_hat'] = (C_readout @ outputs['y_hat'].unsqueeze(-1)).squeeze(-1)
        outputs['y'] = (C_readout @ outputs['y'].unsqueeze(-1)).squeeze(-1)
        plant.y_seq = outputs['y']

    #compute errors
    var_names = ['y'] #['x','a','y']
    err = {err_fn.__name__:
               {v: err_fn(eval(f'plant.{v}_seq'), outputs[f'{v}_hat_ss'], k=30).reshape(shape)
                for v in var_names}
           for err_fn in [mse, bias]}

    if shape == (1,1):
        print(f'R={R:.6g}', 'err:')
        pprint(err)

    err_list.append(err)

#%% Plot controlled dynamics
seq_num = 0#(y_ss == torch.tensor([10.,10])).all(dim=1).nonzero().item()
if plant.dim_x == model.dim_x:
    fig, ax = controller.plot_all(**outputs, x=plant.x_seq, a=plant.a_seq, max_plot_rows=7, seq_num=seq_num)
else:
    fig, ax = controller.plot_all(**outputs, max_plot_rows=7, seq_num=seq_num)


#%%
if shape == (1,1):
    fig, ax = plt.subplots(2,2, sharex=True, figsize=(10,4))
    for r, err_type in enumerate(['mse', 'bias']):
        for c, v in enumerate(var_names):
            err_plot = [err[err_type][v].item() for err in err_list]
            if np.allclose(np.diff(R_list)[:-1], np.diff(R_list)[1:]):
                print(R_list)
                ax[r,c].bar(R_list, err_plot, log=True, width=(R_list[1]-R_list[0])*0.9)
                ax[-1,c].set_xlabel('R')
            else:
                ax[r,c].bar(np.log10(R_list), err_plot, log=True)
                ax[-1,c].set_xlabel('log(R)')
            ax[r,0].set_ylabel(err_type)
            ax[0,c].set_title(f'Variable: {v}' )
    fig.tight_layout()
#%%
K = controller.compute_lqr_gain(Q=Q, R=R)
A,B,C = controller.get_model_matrices()

eigvals = torch.linalg.eigvals(A-B@K)
print(f'eig(A-B@K) = {eigvals[eigvals.abs()>1e-14].numpy()}')
fig, ax = plot_eigvals(A-B@K, title='eig($A-BK$), '+f"$Q={'' if Q==1 else f'{Q}\\times '}C^TC; R={'' if R==1 else f'{R}\\times '}I$")


#%% Plot error heatmaps
err_type = 'mse'

# fix cmap
cmap = mpl.colormaps['Reds']
cmap.set_over('grey')
vmax = defaultdict(lambda: None)
# vmax.update({'x': None, 'a': 17, 'y': None})

#plot
fig, axs = plt.subplots(1,len(var_names), sharex=True, sharey=True, figsize=(4*len(var_names)+1,3.5), squeeze=False)
for var, ax in zip(var_names, axs[0]):
    pcm = ax.pcolormesh(y_ss_0, y_ss_1, err[err_type][var].T,
                        # norm=mpl.colors.LogNorm(vmin=None, vmax=vmax[var]),
                        vmin=None, vmax=vmax[var],
                        cmap=cmap)
    # ax.scatter(*x0[0], c='k')
    # ax.text(*x0[0]+0.15, '$x^{init}$')
    ax.axis('square')
    plt.colorbar(pcm, ax=ax, extend='max' if vmax[var] is not None else None)
    ax.set_title(f'${var}$ {err_type}')
    ax.set_xlabel('Target $y^{ss}_0$')
    ax.set_ylabel('Target $y^{ss}_1$')
# fig.suptitle(f"$Q={'' if Q==1 else f'{Q}\\times '}C^TC; R={'' if R==1 else f'{R}\\times '}I$")
fig.tight_layout()

#%% Histogram to eyeball good values for vmax in heatmaps
# fig, axs = plt.subplots(1, len(var_names), sharex=True, figsize=(13,4))
# for var,ax in zip(var_names, axs):
#     ax.hist(err[var].flatten(), bins=100)
#     ax.set_title(f'{var} errors distribution')
# fig.tight_layout()
