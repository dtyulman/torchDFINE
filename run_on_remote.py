import os, sys, subprocess, datetime, textwrap
from remote_config import REMOTE_SERVER, REMOTE_SYNC_SERVER, REMOTE_PATH, LOCAL_PATH, EMAIL


def submit_remote_job(script_LOCAL_PATH, time='48:00:00', cpus=1, mem=64, partition='main', job_name=''):
    # Partitions on Discovery: https://www.carc.usc.edu/user-guides/hpc-systems/discovery/resource-overview-discovery

    # Rsync command: efficient and checksum-based to avoid unnecessary uploads
    rsync_cmd = ["rsync", "-avvc", "--exclude=results/", "--exclude=old/", "--exclude=.*", "--exclude=__pycache__/", f"{LOCAL_PATH}/", f"{REMOTE_SYNC_SERVER}:{REMOTE_PATH}/"]
    print(f'Syncing...\n {" ".join(rsync_cmd)}')
    subprocess.run(rsync_cmd, check=True)

    # Get relative path of script on remote
    script_name = os.path.basename(script_LOCAL_PATH)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    job_name = f"{os.path.splitext(script_name)[0]}_{timestamp}" + (f"_{job_name}" if job_name else "")
    script_relative_path = os.path.relpath(script_LOCAL_PATH, start=LOCAL_PATH)

    slurm_dir = 'slurm_logs'
    # SLURM batch script (use dedent to make it look nice in the script)
    submit_sh = textwrap.dedent(f"""\
    #!/bin/bash
    #SBATCH --partition={partition}
    #SBATCH --cpus-per-task={cpus}
    #SBATCH --mem-per-cpu={mem}gb
    #SBATCH --time={time}
    #SBATCH --job-name={job_name}
    #SBATCH --output=slurm/%x_%j.out
    #SBATCH --mail-type=END,FAIL
    #SBATCH --mail-user={EMAIL}

    cd {REMOTE_PATH}
    srun python -u {script_relative_path}
    """)

    print("Submitting SLURM job...")
    print(submit_sh)
    submit_cmd = ["ssh", REMOTE_SERVER, "sbatch"]
    subprocess.run(submit_cmd, input=submit_sh, text=True, check=True)
    print(f"Submitted job {job_name} to {REMOTE_SERVER}")
    # sys.exit(0)


def is_running_in_slurm():
    return "SLURM_JOB_ID" in os.environ


def run_me_on_remote(time='48:00:00', cpus=4, mem=48, partition='main', job_name=''):
    """Call this at the top of any script to launch it on the cluster automatically."""
    if not is_running_in_slurm():
        script_LOCAL_PATH = os.path.abspath(sys.argv[0])
        script_name = os.path.basename(script_LOCAL_PATH)
        print(f'Running {script_name} on cluster')
        submit_remote_job(script_LOCAL_PATH, time=time, cpus=cpus, mem=mem, partition=partition, job_name=job_name)
        sys.exit(0)


if __name__ == '__main__':
    script_name = 'control_rnn_mnist.py'
    submit_remote_job(script_name)


#%% Check log
# subprocess.check_call(f'ssh {REMOTE_SERVER} "ls -t {REMOTE_PATH}/slurm/"', shell=True)

# LOG_FILE = 'slurm_AssociativeMNIST_Exceptions_Automatic_35_5504012.out'
# subprocess.check_call(f'ssh {REMOTE_SERVER} "cat {REMOTE_PATH}/slurm/{LOG_FILE}"', shell=True)

#%% Cancel all
# subprocess.check_call(f'ssh {REMOTE_SERVER} "scancel -u {USERNAME}"', shell=True)
