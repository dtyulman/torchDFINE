import subprocess
import itertools

num_gpus = 8
param_dict = dict()
# param_dict['--dim_x'] = list(range(1,11))
param_dict['--dim_x'] = [2]
param_dict['--fit_D_matrix'] = [False,True]
param_dict['--no_manifold'] = [False]
param_dict['--no_input'] = [False,True]
param_dict['--scale_forward_pred'] = [0]
param_dict['--manual_seed'] = [10]

iterables = []
for key in param_dict.keys():
    iterables.append(param_dict[key])

t = list(itertools.product(*iterables))

search_process = []
script_path = 'train_and_control_multi_dim_x.py'
gpu_count = 4
for param_idx, param in enumerate(t):
    search_element = ['/home/ebilgin/oasis_dfine/bin/python3',script_path,'--gpu_id',str(gpu_count%8) if (gpu_count%8) is not 0 else str(gpu_count%8 + 1)]
    gpu_count += 1
    for i in range(len(param)):
        # handle binary arguments seperately
        if list(param_dict.keys())[i] in ['--fit_D_matrix','--no_manifold','--no_input']:
            if param[i]:
                search_element += [list(param_dict.keys())[i]]
        else:
            search_element += ([list(param_dict.keys())[i],str(param[i])])
    print(search_element)
    search_process.append(subprocess.Popen(search_element))


# for process in search_process:
#     process.wait()