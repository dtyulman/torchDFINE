import itertools
import os
import time
import numpy as np

repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
user_name = 'ebilgin'
env_dir = '/scratch1/ebilgin/dfine_env/bin/activate' # path to python file in your venv
time_limit = '48:00:00'
memory_limit = '8GB'
partition ='main'
email = 'ebilgin@usc.edu'
cpus_per_task = 1

bash_txt = f'cd /scratch1/{user_name}/torchDFINE/hpc/slurm\n\n' # main string that includes shell commands
bash_txt += f'export env_dir="{env_dir}"\n' 
script_path = '/scratch1/ebilgin/torchDFINE/scripts/train_and_control_hpc.py' # path of the script that you want to run
slurm_file_path = '/scratch1/ebilgin/torchDFINE/hpc/slurm/slurm_template.slurm'

# bash file name 
time_now = time.localtime()
time_str = time.strftime("%m-%d_%H-%M", time_now)
bash_name = f'{time_str}_jobs_swissroll_simulation.sh'
bash_dir = os.path.join(repo_dir, 'hpc', 'bash'); os.makedirs(bash_dir, exist_ok=True)
log_dir = os.path.join(repo_dir, 'hpc', 'logs'); os.makedirs(log_dir, exist_ok=True)

param_dict = dict()
param_dict['--dim_x'] = np.arange(1,21,1).tolist()
param_dict['--scale_forward_pred'] = np.arange(0,1,.1).tolist()
param_dict['--manual_seed'] = np.arange(10).tolist()
param_dict['--num_seqs'] = [2**16]

param_dict['--fit_D_matrix'] = [False,True]
param_dict['--no_manifold'] = [False,True]
param_dict['--train_ae_seperately'] = [False,True]
param_dict['--train_on_manifold_latent'] = [False,True]

binary_keys = ['--fit_D_matrix','--no_manifold','--train_ae_seperately','--train_on_manifold_latent']

def verify_binary_keys(command):
    command_set = set(command)
    if {'--no_manifold','--train_ae_seperately'}.issubset(command_set) or ({'--train_on_manifold_latent'}.issubset(command_set) and  not {'--train_ae_seperately'}.issubset(command_set)):
        return False
    return True

iterables = []
for key in param_dict.keys():
    iterables.append(param_dict[key])

t = list(itertools.product(*iterables))

for param_idx, param in enumerate(t):
    
    command = ['python3',script_path]
    for i in range(len(param)):
        # handle binary arguments seperately
        if list(param_dict.keys())[i] in binary_keys:
            if param[i]:
                command += [list(param_dict.keys())[i]]
        else:
            command += ([list(param_dict.keys())[i],str(param[i])])

    if verify_binary_keys(command):
        command_str = ' '.join(command) # convert from list to str
        print(command_str)
        command_with_out_dash = [x[2:] if x[:2] == '--' else x for x in command] # remove dashes from parameter names
        job_name_str = '_'.join(command_with_out_dash[2:]) # remove the python command script name
        log_name = f'{job_name_str}.txt'
        bash_txt += f'export log_dir="{log_dir}" log_file="{log_name}";\n'
        bash_txt += f'export command="{command_str}"\n'
        bash_txt += f'sbatch --job-name {job_name_str} --time {time_limit} --mem {memory_limit} --partition {partition} --cpus-per-task {cpus_per_task} --mail-user {email} {slurm_file_path}\n'


with open(os.path.join(bash_dir,bash_name), 'w') as bash_out:
    bash_out.write(bash_txt) 


def verify_binary_keys(command):
    command_set = set(command)
    if {'--no_manifold','--train_ae_seperately'}.issubset(command_set) or ({'--train_on_manifold_latent'}.issubset(command_set) and  not {'--train_ae_seperately'}.issubset(command_set)):
        return False
    return True