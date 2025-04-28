import os
import torch

from PSID import IPSID

from script_utils import get_model
from python_utils import Timer

#%%
# import subprocess
# from remote_config import REMOTE_SYNC_SERVER, REMOTE_PATH, LOCAL_PATH

#NOTE: needed to make the dir first
# rnn_rel_path = 'results/train_logs/2025-04-14/18-29-53_rnn_reach_ksa_OL_y=h_nx=32_ny=32_nu=2/rnn.pt'
# rsync_cmd = ["scp", f"{LOCAL_PATH}/"+rnn_rel_path, f"{REMOTE_SYNC_SERVER}:{REMOTE_PATH}/"+rnn_rel_path]

# print(f'Syncing...\n {" ".join(rsync_cmd)}')
# subprocess.run(rsync_cmd, check=True)

#%%
# from run_on_remote import run_me_on_remote
# run_me_on_remote(mem=0, cpus=32, partition='largemem', job_name='2^17')
# run_me_on_remote(partition='debug')


#%% Plant
dirname = os.path.dirname(os.path.abspath(__file__))
rnn_relpath = '../results/train_logs/2025-04-14/18-29-53_rnn_reach_ksa_OL_y=h_nx=32_ny=32_nu=2/rnn.pt'
rnn_path = os.path.join(dirname, rnn_relpath)

rnn, rnn_train_data = torch.load(rnn_path)
#%% Model training data
data_config = {'num_seqs': 2**14, #2**18
               'num_steps': 50,
               'lo': -0.5,
               'hi': 0.5,
               'levels': 2,
               'include_task_input': False,
               'add_task_input_to_noise': False}
_, _, y, u = rnn_train_data.generate_DFINE_dataset(rnn, **data_config)

y = y.reshape(-1, y.shape[-1]).numpy()
u = u.reshape(-1, u.shape[-1]).numpy()
#%% Model fitting
horizon = 10
dim_x = 32

with Timer('SID'):
    idSys = IPSID(y, U=u, Z=None, n1=0, nx=dim_x, i=horizon)
params = idSys.getListOfParams()

#%%
model_config = {
    'config': {
        'model.dim_x': 32,
        'model.dim_a': rnn.dim_y,
        'model.dim_y': rnn.dim_y,
        'model.dim_u': rnn.dim_u,
        'model.hidden_layer_list': None,
        }
    }

save_str = (f"_rnn_reach_psid"
            f"_u={data_config['lo']}-{data_config['hi']}-{data_config['levels']}"
            f"_y={rnn.obs_fn.obs}"
            f"_nx={model_config['config']['model.dim_x']}"
            f"_ny={model_config['config']['model.dim_y']}"
            f"_nu={model_config['config']['model.dim_u']}")
model_config['config']['savedir_suffix'] = save_str

config, trainer = get_model(**model_config)

trainer.dfine.ldm.A.data = torch.from_numpy(idSys.A).to(torch.float32)
trainer.dfine.ldm.C.data = torch.from_numpy(idSys.C).to(torch.float32)
trainer.dfine.ldm.B.data = torch.from_numpy(idSys.B).to(torch.float32)
trainer.dfine.ldm.W_log_diag.data = torch.diag(torch.log(torch.from_numpy(idSys.Q))).to(torch.float32)
trainer.dfine.ldm.R_log_diag.data = torch.diag(torch.log(torch.from_numpy(idSys.R))).to(torch.float32)

trainer._save_ckpt(epoch=1, model=trainer.dfine, optimizer=trainer.optimizer, lr_scheduler=trainer.lr_scheduler)
