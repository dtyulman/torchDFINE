import torch
import torch.nn as nn 
from tqdm import tqdm

class TrainerAutoencoder():
    def __init__(self,**kwargs):
        self.model = kwargs.pop('model')
        self.lr = kwargs.pop('lr')
        self.batch_size = kwargs.pop('batch_size')
        self.num_epochs = kwargs.pop('num_epochs')
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(),lr=self.lr)

    def train_epoch(self,epoch,train_loader):
        with tqdm(train_loader, unit='batch') as tepoch:
           for batch in train_loader:
                tepoch.set_description(f"Epoch {epoch}, TRAIN")

                # Carry data to device
                # batch = carry_to_device(data=batch, device=self.device)
                y_batch, _, _, _ = batch

                outs = self.model(y_batch.view(-1,y_batch.shape[-1]))
                loss = self.criterion(outs, y_batch.view(-1,y_batch.shape[-1]))
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                print(f'train loss: {loss.item()}')

    def valid_epoch(self,epoch,valid_loader):
        with tqdm(valid_loader, unit='batch') as tepoch:
           for batch in valid_loader:
                tepoch.set_description(f"Epoch {epoch}, VALID")

                # Carry data to device
                # batch = carry_to_device(data=batch, device=self.device)
                y_batch, _, _, _ = batch

                outs = self.model(y_batch.view(-1,y_batch.shape[-1]))
                loss = self.criterion(outs, y_batch.view(-1,y_batch.shape[-1]))
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        
    def train(self,train_loader,valid_loader=None):
        for epoch in range(1,self.num_epochs+1):
            self.train_epoch(epoch,train_loader)

            if isinstance(valid_loader,torch.utils.data.dataloader.DataLoader):
                self.valid_epoch(epoch,valid_loader)