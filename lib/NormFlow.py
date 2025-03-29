# Standard library imports
import torch
import pytorch_lightning as pl
from .NF_comp import NormalizingQuadFlow


class NormalizingFlowModel(pl.LightningModule):
    """
    PyTorch Lightning Module for training the normalizing flow.
    """
    def __init__(self, input_dim=2, n_layers=10, lr=1e-3,X_mean=1,X_std=0,c=1,clamp_val=-10):
        super().__init__()
        self.save_hyperparameters()
        self.flow = NormalizingQuadFlow(input_dim, n_layers)

        self.lr = lr
        self.X_mean = torch.tensor(X_mean,dtype=torch.float32).to(self.device)
        self.X_std = torch.tensor(X_std,dtype=torch.float32).to(self.device)
        self.prior = torch.distributions.normal.Normal(loc=0.0, scale=1.0)
        self.c = c
        self.train_losses = []
        self.val_losses = []
        self.clamp_val = clamp_val
        self.input_dim = input_dim

    def forward(self, x,eval=True):
        if eval:
            with torch.no_grad():
                x = (x - self.X_mean) / (self.X_std + 1e-8)  # small epsilon to avoid divide-by-zero
                z, log_det = self.flow(x)
                
                # Replace NaN or Inf with 0.0 just to avoid the error
                z = torch.nan_to_num(z, nan=0.0, posinf=1e3, neginf=-1e3)
                
                log_z = self.prior.log_prob(z).sum(dim=1)
                log_prob = log_z + log_det
        else:
            x = (x - self.X_mean) / (self.X_std + 1e-8)  # small epsilon to avoid divide-by-zero
            z, log_det = self.flow(x)
            log_z = self.prior.log_prob(z).sum(dim=1)
            log_prob = log_z + log_det

        return log_prob

    
    def training_step(self, batch, batch_idx):
        if len(batch)>1:
            x,y = batch
            log_prob = self.forward(x,eval=False)
            log_prob_adv = self.forward(y,eval=False)

            log_prob_adv = torch.clamp(log_prob_adv,min=self.clamp_val )
            loss = -self.c*log_prob.mean()+log_prob_adv.mean()
            self.log("train_logprob_adv", log_prob_adv.mean(), prog_bar=True)

        else:
            x = batch[0]
            log_prob = self.forward(x)
            loss = -log_prob.mean()

        #self.log("train_varloss", var_loss.mean(), prog_bar=True)
        self.log("train_logprob", log_prob.mean(), prog_bar=True)
        self.log("train_loss", loss, prog_bar=True)
        self.train_losses.append(loss)

        return loss
    
    def validation_step(self, batch, batch_idx):
        if len(batch)>1:
            x,y = batch
            log_prob = self.forward(x,eval=False)
            log_prob_adv = self.forward(y,eval=False)

            log_prob_adv = torch.clamp(log_prob_adv,min=self.clamp_val  )
            loss = -self.c*log_prob.mean()+log_prob_adv.mean()
        else:
            x = batch[0]
            log_prob = self.forward(x)
            loss = -log_prob.mean()
        self.log("val_loss", loss, prog_bar=True)
        self.val_losses.append(loss)
        return loss
    
        
    def sample(self, num_samples,grad=False):
        """
        Sample from the learned distribution.
        1. Sample latent variable z from the base distribution.
        2. Apply the inverse flow to obtain samples in data space.
        """
        # Sample from base distribution. Shape: (num_samples, 1)
        z = self.base_distribution.sample((num_samples,))

        if grad:
            x_samples = self.flow.inverse(z)
        else:
            # Transform to data space.
            with torch.no_grad():
                x_samples = self.flow.inverse(z)
        
        x_samples = (x_samples*self.X_std)+self.X_mean
        return x_samples
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)