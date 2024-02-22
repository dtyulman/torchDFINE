import os
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.ssm import NonlinearStateSpaceModel, make_nonlinear_embedding_fn, plot_z, plot_y
from datasets import DFINEDataset
from trainers.TrainerDFINE import TrainerDFINE
from config_dfine import get_default_config

#%% Make a ground-truth SSM
dim_z = 1
dim_u = 1
dim_a = 1
dim_y = 3

A = np.array([[0.99]])
B = np.array([[1]])
C = np.eye(dim_z)
f = make_nonlinear_embedding_fn(dim_y, manifold_type='ring')

Q = 0 * np.diag(np.random.rand(dim_z))
R = np.zeros((dim_a, dim_a))
S = 0.01 * np.diag(np.random.rand(dim_y))

ssm = NonlinearStateSpaceModel(A,B,C,f, Q,R,S)


#%%
config = get_default_config()
config.model.dim_x = dim_z
config.model.dim_u = dim_u
config.model.dim_a = dim_a
config.model.dim_y = dim_y


#%% Generate data 
num_steps = 100
u = np.ones((num_steps, dim_u)) * 0.2
z0 = np.zeros(dim_z)
z,a,y = ssm.generate_trajectory(z0=z0, u_seq=u)
# plot_z(z)
# plot_y(y)

train_data = DFINEDataset(torch.tensor(y, dtype=torch.float32).unsqueeze(0))#, torch.tensor(u, dtype=torch.float32).unsqueeze(0)) #add batch dimension 
train_loader = DataLoader(train_data, batch_size=config.train.batch_size) 


#%% Train DFINE

trainer = TrainerDFINE(config)
trainer.train(train_loader)
#%% Plot 

