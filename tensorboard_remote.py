#!/usr/bin/env python

"""Run tensorboard on remote, port forward, and open in browser"""
import os, subprocess, time, webbrowser, random, threading, argparse, datetime
import urllib.request
from remote_config import REMOTE_SERVER, REMOTE_PATH, USERNAME

parser = argparse.ArgumentParser(description="Launch TensorBoard on cluster via Slurm.")
parser.add_argument('--date', type=str, default=None, help="Date string in YYYY-MM-DD format. Defaults to today.")
args = parser.parse_args()
if args.date is None:
    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
else:
    date_str = args.date

REMOTE_LOGDIR = os.path.join(REMOTE_PATH, 'results', 'train_logs', date_str)
TENSORBOARD_LOGFILE = 'tensorboard.out'
TENSORBOARD_PORT = random.randint(15000, 25000) #random high port to avoid conflicts
LOCAL_PORT = TENSORBOARD_PORT  #same port locally


#Build the SLURM script
slurm_script = f"""#!/bin/bash
#SBATCH --job-name=tensorboard
#SBATCH --partition=main
#SBATCH --time=8:00:00
#SBATCH --output={TENSORBOARD_LOGFILE}

tensorboard --logdir {REMOTE_LOGDIR} --port {TENSORBOARD_PORT} --host 127.0.0.1
"""

#Submit SLURM job
print(f"Opening TensorBoard at {REMOTE_SERVER}:{REMOTE_LOGDIR}")
submit_cmd = ["ssh", REMOTE_SERVER, "sbatch"]
submit_proc = subprocess.run(submit_cmd, input=slurm_script, text=True, capture_output=True, check=True)
job_id = submit_proc.stdout.strip().split()[-1]
print(f"Submitted SLURM job ID: {job_id}")

#Poll for compute node
print("Waiting for job to start and node to be assigned...")
node_name = None
while node_name is None:
    time.sleep(5)
    squeue_cmd = ["ssh", REMOTE_SERVER, f"squeue -j {job_id} -o '%N' --noheader"]
    try:
        node_name = subprocess.check_output(squeue_cmd, text=True).strip()
        if node_name in ("", "None assigned", "n/a"):
            node_name = None
    except subprocess.CalledProcessError:
        continue
print(f"Job is running on node: {node_name}")

#Set up SSH tunnels
print("Starting SSH tunnel: local → login → compute...")
local_to_login = subprocess.Popen(["ssh", "-N", "-L", f"{LOCAL_PORT}:localhost:{LOCAL_PORT}", REMOTE_SERVER])
login_to_compute = subprocess.Popen(["ssh", REMOTE_SERVER, f"ssh -N -L {LOCAL_PORT}:localhost:{TENSORBOARD_PORT} {USERNAME}@{node_name}"])

#Start log tailing (non-buffered, line-by-line)
print("Starting live log stream...")
tail_cmd = ["ssh", REMOTE_SERVER, f"tail -n +1 -F {TENSORBOARD_LOGFILE}"]
tail_proc = subprocess.Popen(tail_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)

def stream_tail_output(proc):
    for line in proc.stdout:
        print(line, end='', flush=True)
stream_thread = threading.Thread(target=stream_tail_output, args=(tail_proc,), daemon=True)
stream_thread.start()


#Open browser
url = f"http://localhost:{LOCAL_PORT}"
print(f"Waiting for TensorBoard to become available at {url}...")
timeout_seconds = 60
start_time = time.time()
while True:
    try:
        urllib.request.urlopen(url, timeout=1)
        print("TensorBoard is ready. Opening browser...")
        webbrowser.open(url)
        break
    except Exception:
        if time.time() - start_time > timeout_seconds:
            print(f"Error: TensorBoard did not start within {timeout_seconds} seconds.")
            break
        time.sleep(1)


#Cleanup
try:
    print("Tunnels established. Live logs are streaming below. Press Ctrl+C to terminate.")
    while True:
        time.sleep(2)
        if tail_proc.poll() is not None:
            print("\n[tail] Log stream ended.")
            break
        if local_to_login.poll() is not None:
            print("\n[ssh] Tunnel local→login closed.")
            break
        if login_to_compute.poll() is not None:
            print("\n[ssh] Tunnel login→compute closed.")
            break
except KeyboardInterrupt:
    print("\nManual interrupt received.")
finally:
    print("Cleaning up...")
    # Close tunnels and log stream
    for proc in [local_to_login, login_to_compute, tail_proc]:
        if proc.poll() is None:
            proc.terminate()
    stream_thread.join(timeout=5)

    # Cancel SLURM job
    try:
        subprocess.run(["ssh", REMOTE_SERVER, f"scancel {job_id}"], check=False)
        print(f"Cancelled SLURM job {job_id}.")
    except Exception as e:
        print(f"Warning: Failed to cancel SLURM job {job_id}: {e}")
