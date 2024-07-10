import itertools
import os
import time

repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
user_name = 'ebilgin'
python_dir = '/home/ebilgin/oasis_dfine/bin/python3' # path to python file in your venv
time_limit = '48:00:00'
memory_limit = '8GB'
partition ='main'
email = 'ebilgin@usc.edu'
cpus_per_task = 1

bash_txt = f'cd /scratch1/{user_name}/torchDFINE/hpc/slurm\n\n' # main string that includes shell commands 
script_path = 'scripts/train_and_control_multi_dim_x.py' # relative path of the script that you want to run
slurm_file_path = 'slurm_template.slurm'

# bash file name 
time_now = time.localtime()
time_str = time.strftime("%m-%d_%H-%M", time_now)
bash_name = f'{time_str}_jobs_swissroll_simulation.sh'
bash_dir = os.path.join(repo_dir, 'hpc', 'bash'); os.makedirs(bash_dir, exist_ok=True)
# log_dir = os.path.join(repo_dir, 'hpc', 'logs'); os.makedirs(bash_dir, exist_ok=True)

param_dict = dict()
param_dict['--dim_x'] = [2,3,4]
param_dict['--fit_D_matrix'] = [False]
param_dict['--no_manifold'] = [False]
# param_dict['--scale_forward_pred'] = [0]
# param_dict['--manual_seed'] = [10]


iterables = []
for key in param_dict.keys():
    iterables.append(param_dict[key])

t = list(itertools.product(*iterables))

for param_idx, param in enumerate(t):
    
    command = [python_dir,script_path]
    for i in range(len(param)):
        # handle binary arguments seperately
        if list(param_dict.keys())[i] in ['--fit_D_matrix','--no_manifold']:
            if param[i]:
                command += [list(param_dict.keys())[i]]
        else:
            command += ([list(param_dict.keys())[i],str(param[i])])

    command_str = ' '.join(command) # convert from list to str
    command_with_out_dash = [x[2:] if x[:2] == '--' else x for x in command] # remove dashes from parameter names
    job_name_str = '_'.join(command_with_out_dash[2:]) # remove the python command script name
    bash_txt += f'export command="{command_str}"\n'
    bash_txt += f'sbatch --job-name {job_name_str} --time {time_limit} --mem {memory_limit} --partition {partition} --cpus_per_task {cpus_per_task} --mail-user {email} {slurm_file_path}\n'


with open(os.path.join(bash_dir,bash_name), 'w') as bash_out:
    bash_out.write(bash_txt) 
