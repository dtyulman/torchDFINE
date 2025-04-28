import os

"""
Discovery cluster docs: https://www.carc.usc.edu/user-guides/hpc-systems/discovery
"""

USERNAME = 'tyulmank'
EMAIL = f'{USERNAME}@usc.edu'
REMOTE_SERVER = f'{USERNAME}@discovery.usc.edu' #must have ssh keys set up https://www.hostinger.com/tutorials/ssh/how-to-set-up-ssh-keys
REMOTE_SYNC_SERVER = f'{USERNAME}@discovery.usc.edu' #f'{USERNAME}@hpc-transfer1.usc.edu' makes me dual authenticate every time >:(
REMOTE_PATH = f'/home1/{USERNAME}/torchDFINE'
LOCAL_PATH = os.path.dirname(os.path.abspath(__file__))
