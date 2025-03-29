# Standard library imports
import torch
from torch import nn
import pytorch_lightning as pl
import torch.nn.functional as F

class CombinedClassifier(pl.LightningModule):
    def __init__(self, hidden_dim=128, latent_dim=256, lr=1e-3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.lr = lr
        self.save_hyperparameters()
        
        # Lists to track training and validation losses
        self.train_loss_list = []
        self.val_loss_list = []
        
        # Separate input layers for each category
        self.input_2j = nn.Sequential(
            nn.Linear(31, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.input_1j = nn.Sequential(
            nn.Linear(24, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # # Uncomment and modify if you need a head for jet_category 0
        # self.input_0j = nn.Sequential(
        #     nn.Linear(18, hidden_dim),
        #     nn.BatchNorm1d(hidden_dim),
        #     nn.GELU(),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.GELU(),
        #     nn.Linear(hidden_dim, hidden_dim)
        # )

        # Shared trunk after the initial layers
        self.shared_trunk = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim), 
            nn.GELU()
        )

        # Classification heads for each category
        self.classifier_2j = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, 1)  # BCE output (logits)
        )
        self.classifier_1j = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, 1)
        )
        # # Uncomment if needed for jet_category 0
        # self.classifier_0j = nn.Sequential(
        #     nn.Linear(latent_dim, latent_dim),
        #     nn.GELU(),
        #     nn.Linear(latent_dim, 1)
        # 
    
    def forward(self, x, jet_category):
        """
        x: [batch_size, N, feature_dim] for the given jet_category
        jet_category in {1,2} indicates which input head to use
        Returns:
            class_logits: [batch_size, N, 1]
        """
        if jet_category == 2:
            x = self.input_2j(x)
        elif jet_category == 1:
            x = self.input_1j(x)
        # else:
        #     x = self.input_0j(x)

        # Shared trunk
        x = self.shared_trunk(x)

        # Classification
        if jet_category == 2:
            logits = self.classifier_2j(x)
        elif jet_category == 1:
            logits = self.classifier_1j(x)
        # else:
        #     logits = self.classifier_0j(x)
        return logits

    def training_step(self, batch, batch_idx):
        """
        Expects batch to be a tuple: (x, jet_category, y)
          - x: input tensor
          - jet_category: integer (1 or 2) to choose the proper branch
          - y: target tensor (same shape as logits)
        """
        x_2j  = batch["x_2j"]
        x_1j  = batch["x_1j"]
        #x_0j  = batch["x_0j"]
        l_2j  = batch["l_2j"].float()
        l_1j  = batch["l_1j"].float()
        #l_0j  = batch["l_0j"].float()

        logits_2j = self.forward(x_2j, 2).squeeze(1)
        logits_1j = self.forward(x_1j, 1).squeeze(1)
        #logits_0j = self.forward(x_0j, 0).squeeze(1)

        loss = F.binary_cross_entropy_with_logits(logits_2j, l_2j)+ F.binary_cross_entropy_with_logits(logits_1j, l_1j)

        self.train_loss_list.append(loss.item())
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x_2j  = batch["x_2j"]
        x_1j  = batch["x_1j"]
        #x_0j  = batch["x_0j"]
        l_2j  = batch["l_2j"].float()
        l_1j  = batch["l_1j"].float()
        #l_0j  = batch["l_0j"].float()

        logits_2j = self.forward(x_2j, 2).squeeze(1)
        logits_1j = self.forward(x_1j, 1).squeeze(1)
        #logits_0j = self.forward(x_0j, 0).squeeze(1)

        loss = F.binary_cross_entropy_with_logits(logits_2j, l_2j)+ F.binary_cross_entropy_with_logits(logits_1j, l_1j)

        self.val_loss_list.append(loss.item())
        self.log('val_loss', loss,prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer