from einops import rearrange
import pytorch_lightning as pl
import wandb
import numpy as np
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
import datetime
from transformer import Transformer
import sys
sys.path.append("..")
import pickle as pkl


class AdaptationNetwork(LightningModule):
    def __init__(
        self,
        FLAGS,
        # config_wandb,
        **kwargs,
    ):
        super().__init__()
        self.FLAGS = FLAGS

        input_channels = 2

        self.encoder = nn.Sequential(
            nn.Linear(input_channels, 128),
            nn.LeakyReLU(),
            Transformer(128, 6, 16, 64, 128, 0.1),
            nn.Linear(128, input_channels)
        )
        self.can_plot = True
        self.new_joints = []
        self.inputs = []
        self.pos_enc = nn.Parameter(torch.randn(1,2*self.FLAGS.n_joints), requires_grad=True)

    def forward(self, z, FLAGS): 
        # facciamo due prove: mascheriamo solo gli sci, impara solo a ricostruire solo gli sci quindi avra sempre bisogno dello scheletro di yolo completo
        # forse emglio fare rand index cosi che impari tutto e nel caso yolo non stimasse bene tutte le joints possiamo usare questo
        """
        z shape: (Batch, Joints, 2)
        """
        mask_indices = []

        if FLAGS.mode == 'train':
            if FLAGS.masking_mode == 'zero' or FLAGS.masking_mode == 'random':
                # Sceglie N joint a caso da mascherare
                rand_index = np.sort(np.random.choice(range(0, self.FLAGS.n_joints), self.FLAGS.masked_joints, replace=False))
                mask_indices = torch.tensor(rand_index, device=z.device).long()
                
                if FLAGS.masking_mode == 'zero':
                    z[:, mask_indices, :] = 0.0
                elif FLAGS.masking_mode == 'random':
                    z[:, mask_indices, :] = torch.randn_like(z[:, mask_indices, :]) * 0.1
            elif FLAGS.masking_mode == 'end_effectors':
                # ATTENZIONE: Questi indici devono corrispondere a Mani/Piedi nel TUO dataset (23 joints)
                indices = [5, 8, 12, 19]
               
                try:
                    mask_indices = torch.tensor(indices, device=z.device).long()
                    z[:, mask_indices, :] = 0.0
                except:
                    pass

            if not isinstance(mask_indices, torch.Tensor):
                mask_indices = torch.tensor([], device=z.device).long()

        b, j, d = z.shape
        z_flat = rearrange(z, 'b j d -> b (d j)')

        if z_flat.shape[1] == self.pos_enc.shape[1]:
            z_flat = z_flat + self.pos_enc
        else:
             print(f"Warning: PosEnc shape {self.pos_enc.shape} != Input shape {z_flat.shape}")

        z_reshaped = rearrange(z_flat, 'b (d j) -> b j d', d=2)
        
        output = self.encoder(z_reshaped)
        
        return output, mask_indices
    

    def training_step(self, batch, batch_idx):

        inputs = batch['input_2d']

        x = inputs[:, :, :2].clone()
        target = x.clone()

        reconstructed, mask_idx = self(x, self.FLAGS)

        mse_loss = nn.MSELoss()
        if len(mask_idx) > 0:
            loss = mse_loss(reconstructed[:, mask_idx, :], target[:, mask_idx, :])
        else:
            loss = mse_loss(reconstructed, target)

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs = batch['input_2d']
        x = inputs[:, :, :2].clone()
        target = x.clone()

        reconstructed, mask_idx = self(x, self.FLAGS)

        mse_loss = nn.MSELoss()
        if len(mask_idx) > 0:
            loss = mse_loss(reconstructed[:, mask_idx, :], target[:, mask_idx, :])
        else:
            loss = mse_loss(reconstructed, target)
            
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        inputs = batch['input_2d']
        x = inputs[:, :, :2].clone()
        target = x.clone()
        
        reconstructed, _ = self(x, self.FLAGS)
        
        test_loss = nn.MSELoss()(reconstructed, target)
        self.log("test_loss", test_loss)
        
        # Qui potresti salvare le immagini o i plot se vuoi vedere i risultati qualitativi

    def configure_optimizers(self):
        lr = self.FLAGS.lr

        opt_encoder = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        scheduler_encoder = torch.optim.lr_scheduler.ExponentialLR(optimizer=opt_encoder, gamma=0.95)

        return [opt_encoder], [scheduler_encoder]