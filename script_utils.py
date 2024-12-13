import timeit

import torch
from torch.utils.data import DataLoader

from data.SSM import make_dfine_from_ssm
from config_dfine import make_config, load_config
from trainers.TrainerDFINE import TrainerDFINE


class Timer:
    """http://preshing.com/20110924/timing-your-code-using-pythons-with-statement/"""
    def __init__(self, name='Timer', verbose=True):
        self.name = name
        self.verbose = verbose

    def __enter__(self):
        if self.verbose:
            print( 'Starting {}...'.format(self.name) )
        self.start = timeit.default_timer()
        return self

    def __exit__(self, *args):
        self.end = timeit.default_timer()
        self.elapsed = self.end - self.start
        if self.verbose:
            print( '{}: elapsed {} sec'.format(self.name, self.elapsed) )



def get_model(config=None, train_data=None, load_path=None, ckpt=None, ground_truth=None):
    """
    a) Load a model. Specify model_path and optionally ckpt
    b) Create a model using parameters from NonlinearStateSpaceModel. Specify NLSSM instance.
    c) Train a new model. Specify config dict which optionally includes savedir_suffix.
    """
    if load_path: #get_model(load_path, ckpt)
        assert config is None and ground_truth is None
        config = load_config(load_path, ckpt)
        trainer = TrainerDFINE(config)

    elif ground_truth: #get_model(ground_truth=ssm)
        assert load_path is None and ckpt is None and config is None
        config = {'model.dim_x': ground_truth.dim_x,
                  'model.dim_u': ground_truth.dim_u,
                  'model.dim_a': ground_truth.dim_a,
                  'model.dim_y': ground_truth.dim_y,
                  'train.num_epochs': 0}
        config = make_config(**config)
        trainer = TrainerDFINE(config)
        trainer.dfine = make_dfine_from_ssm(ground_truth, trainer.dfine)

    else: #get_model(config, train_data)
        assert load_path is None and ckpt is None and ground_truth is None
        config = make_config(**config)
        trainer = TrainerDFINE(config)
        train_loader = DataLoader(train_data, batch_size=config.train.batch_size)
        try:
            trainer.train(train_loader)
        except KeyboardInterrupt: #return model in its current state if interrupt training
            return config, trainer

    return config, trainer
