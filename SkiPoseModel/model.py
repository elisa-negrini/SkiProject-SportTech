from einops import rearrange
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
import sys
import pickle as pkl
from transformer import Transformer
import random

class AdaptationNetwork(LightningModule):
    def __init__(self, FLAGS, **kwargs):
        super().__init__()
        self.FLAGS = FLAGS
        self.save_hyperparameters(ignore=['FLAGS'])

        self.ski_joints = FLAGS.ski_joints if isinstance(FLAGS.ski_joints, list) else list(map(int, FLAGS.ski_joints))

        self.encoder = nn.Sequential(
            nn.Linear(2, 128),  
            nn.LeakyReLU(),    
            Transformer(128, 6, 16, 64, 128, 0.), 
            nn.Linear(128, 2)  
        )

        self.pos_enc = nn.Parameter(torch.randn(1, 2 * self.FLAGS.n_joints), requires_grad=True)

        self.test_preds = []
        self.test_gts = []

    def forward(self, z, FLAGS): 
        """
        z shape: (Batch, 23, 2)
        """
        mask_indices = []
        
        if FLAGS.mode == 'train':
            mask_indices = self.ski_joints
            
            mask_indices_tensor = torch.tensor(mask_indices, device=z.device).long()
            z[:, mask_indices_tensor, :] = 0.0
            
        elif FLAGS.mode in ['demo', 'test']:
            mask_indices = self.ski_joints
            mask_indices_tensor = torch.tensor(mask_indices, device=z.device).long()
            z[:, mask_indices_tensor, :] = 0.0
        
        b, j, d = z.shape
        z_flat = rearrange(z, 'b j d -> b (d j)')

        if z_flat.shape[1] == self.pos_enc.shape[1]:
            z_flat = z_flat + self.pos_enc
        
        z_reshaped = rearrange(z_flat, 'b (d j) -> b j d', d=2)

        output = self.encoder(z_reshaped)

        return output, mask_indices

    def training_step(self, batch, batch_idx):
        inputs = batch['input_2d']
        target = inputs.clone()
        
        full_pred, mask_idx = self(inputs.clone(), self.FLAGS)
    
        pred_masked = full_pred[:, mask_idx, :]
        gt_masked = target[:, mask_idx, :]
        
        loss = nn.MSELoss()(pred_masked, gt_masked)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs = batch['input_2d']
        target = inputs.clone()
        
        full_pred, _ = self(inputs.clone(), self.FLAGS)
        
        ski_indices = torch.tensor(self.ski_joints, device=target.device).long()
        ski_pred = full_pred[:, ski_indices, :]
        ski_gt = target[:, ski_indices, :]
        
        loss_ski = nn.MSELoss()(ski_pred, ski_gt)
        self.log("val_loss", loss_ski, prog_bar=True)
        return loss_ski

    def test_step(self, batch, batch_idx):
        inputs = batch['input_2d']
        target = inputs.clone()
        
        full_pred, _ = self(inputs.clone(), self.FLAGS)
        
        final_result = target.clone()
        ski_indices = torch.tensor(self.ski_joints, device=target.device).long()
        final_result[:, ski_indices, :] = full_pred[:, ski_indices, :]

        ski_loss = nn.MSELoss()(full_pred[:, ski_indices, :], target[:, ski_indices, :])
        
        self.log("test_loss_ski", ski_loss, on_epoch=True)
    
        self.test_preds.append(final_result.detach().cpu())
        self.test_gts.append(target.detach().cpu()) 
        
        return ski_loss

    def on_test_epoch_end(self):
        if len(self.test_preds) > 0:
            all_preds = torch.cat(self.test_preds, dim=0).float().numpy()
            all_gts = torch.cat(self.test_gts, dim=0).float().numpy()
            
            save_path = "test_results.pkl"
            with open(save_path, 'wb') as f:
                pkl.dump({'pred': all_preds, 'gt': all_gts}, f)
                
            print(f"File saved: {save_path} with {all_preds.shape[0]} samples.")
        else:
            print("⚠️ No prediction saved!")
        
        self.test_preds = []
        self.test_gts = []

    def configure_optimizers(self):
        lr = self.FLAGS.lr 
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5)
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sched,
                "monitor": "val_loss"
            }
        }