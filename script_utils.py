import os
import timeit

import torch
from torch.utils.data import DataLoader

import data.RNN as RNN
import data.SSM as SSM
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
        # Load the model and its config from load_path
        assert config is None and ground_truth is None
        config = load_config(load_path, ckpt)
        trainer = TrainerDFINE(config)

    elif ground_truth: #get_model(ground_truth=ssm)
        assert load_path is None and ckpt is None and config is None

        if isinstance(ground_truth, SSM.NonlinearStateSpaceModel):
            config = {'model.dim_x': ground_truth.dim_x,
                      'model.dim_u': ground_truth.dim_u,
                      'model.dim_a': ground_truth.dim_a,
                      'model.dim_y': ground_truth.dim_y,
                      'train.num_epochs': 0}
            config = make_config(**config)
            trainer = TrainerDFINE(config)
            trainer.dfine = SSM.make_dfine_from_ssm(ground_truth, trainer.dfine)
        elif isinstance(ground_truth, RNN.RNN):
            config = {'model.dim_x': ground_truth.dim_h,
                      'model.dim_u': ground_truth.dim_u,
                      'model.dim_a': ground_truth.dim_y,
                      'model.dim_y': ground_truth.dim_y,
                      'train.num_epochs': 0}
            config = make_config(**config)
            trainer = TrainerDFINE(config)
            trainer.dfine = RNN.make_dfine_from_rnn(ground_truth, trainer.dfine)


    else: #get_model(config, train_data)
        # Train the model from scratch
        assert load_path is None and ckpt is None and ground_truth is None

        #ensure that there is no bottleneck in y->a transformation
        assert all([nl >= config['model.dim_a'] for nl in config['model.hidden_layer_list']])

        config = make_config(**config)
        trainer = TrainerDFINE(config)

        if train_data is not None:
            train_loader = DataLoader(train_data, batch_size=config.train.batch_size, shuffle=True)
            try:
                trainer.train(train_loader)
            except KeyboardInterrupt: #return model in its current state if interrupt training
                return config, trainer

    return config, trainer



def resume_training(trainer, train_data, start_epoch=-1, additional_epochs=1, batch_size=None, override_save_dir=None, override_lr=None):
    from torch.utils.tensorboard import SummaryWriter

    #fix save dir
    assert os.path.isdir(trainer.config.model.save_dir)
    if override_save_dir:
        trainer.config.model.save_dir = override_save_dir
        trainer.ckpt_save_dir = os.path.join(trainer.config.model.save_dir, 'ckpts')
        trainer.plot_save_dir = os.path.join(trainer.config.model.save_dir, 'plots')
        trainer.writer = SummaryWriter(log_dir=os.path.join(trainer.config.model.save_dir, 'summary'))

    #fix LR
    if override_lr:
        trainer.config.lr.init = override_lr
        trainer.optimizer = trainer._get_optimizer(params=trainer.dfine.parameters())
        trainer.lr_scheduler = trainer._get_lr_scheduler()

    #extend epochs
    trainer.start_epoch = trainer.config.train.num_epochs+1 if start_epoch == -1 else start_epoch
    trainer.config.train.num_epochs = trainer.start_epoch + additional_epochs

    #make dataloader
    trainer.dfine.requires_grad_()
    train_loader = DataLoader(train_data, batch_size=batch_size or trainer.config.train.batch_size)
    trainer.train(train_loader)
