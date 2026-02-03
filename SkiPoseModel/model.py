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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def straighten_ski_points(points):
    """
    Straightens 4 ski points by projecting them onto the best-fit line.
    Fully differentiable with PyTorch.
    """
    original_dtype = points.dtype
    points = points.float()
    
    # points: (B, 4, 2)
    B = points.shape[0]
    centroid = points.mean(dim=1, keepdim=True)  # (B, 1, 2)
    points_centered = points - centroid  # (B, 4, 2)
    P = points_centered.transpose(1, 2)  # (B, 2, 4)
    
    cov = torch.bmm(P, P.transpose(1, 2)) / 4.0  # (B, 2, 2)
    
    # SVD to find principal direction (eigenvector with max eigenvalue)
    # cov = U @ S @ V.T
    U, S, V = torch.svd(cov)  # U: (B, 2, 2)
    # Principal dir
    direction = U[:, :, 0]  # (B, 2)

    # Project every point onto the line passing through the centroid
    # Formula: proj = centroid + ((p - centroid) · direction) * direction
    direction = direction.unsqueeze(1)  # (B, 1, 2)
    dot_products = (points_centered * direction).sum(dim=2, keepdim=True)  # (B, 4, 1)
    projections = centroid + dot_products * direction  # (B, 4, 2)
    
    projections = projections.to(original_dtype)
    return projections

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
        self.test_preds = []
        self.test_gts = []
        self.inputs = []
        self.pos_enc = nn.Parameter(torch.randn(1,2*self.FLAGS.n_joints), requires_grad=True)

    def forward(self, z, FLAGS): 
        """
        z shape: (Batch, Joints, 2)
        """
        mask_indices = []

        if FLAGS.mode == 'train':
            ##### mask zeros
            # ski_joints = [19, 20, 21, 22, 12, 13, 14, 15]
            # mask_indices = torch.tensor(ski_joints, device=z.device).long()
            # z[:, mask_indices, :] = 0.0

            ##### mask on ankles
            # Left Ski (12,13,14,15) -> initialized to ankle 11
            z[:, [12,13,14,15], :] = z[:, [11], :].repeat(1, 4, 1)
            # Right Ski (19,20,21,22) -> initialized to ankle 18
            z[:, [19,20,21,22], :] = z[:, [18], :].repeat(1, 4, 1)
            mask_indices = torch.tensor([19, 20, 21, 22, 12, 13, 14, 15], device=z.device).long()
                
        elif FLAGS.mode == 'demo' or FLAGS.mode == 'test':
            ##### mask zeros
            # ski_joints = [19, 20, 21, 22, 12, 13, 14, 15]
            # mask_indices = torch.tensor(ski_joints, device=z.device).long()
            # z[:, mask_indices, :] = 0.0

            ##### mask on ankles
            # Left Ski (12,13,14,15) -> initialized to ankle 11
            z[:, [12,13,14,15], :] = z[:, [11], :].repeat(1, 4, 1)
            # Right Ski (19,20,21,22) -> initialized to ankle 18
            z[:, [19,20,21,22], :] = z[:, [18], :].repeat(1, 4, 1)
            mask_indices = torch.tensor([19, 20, 21, 22, 12, 13, 14, 15], device=z.device).long()

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
        
        ski_indices_dx = torch.tensor([19, 20, 21, 22], device=reconstructed.device).long()
        ski_indices_sx = torch.tensor([12, 13, 14, 15], device=reconstructed.device).long()
        
        ski_pred_dx = reconstructed[:, ski_indices_dx, :]
        ski_pred_sx = reconstructed[:, ski_indices_sx, :]
        
        ski_gt_dx = target[:, ski_indices_dx, :]
        ski_gt_sx = target[:, ski_indices_sx, :]
        
        ski_pred_dx_straight = straighten_ski_points(ski_pred_dx)
        ski_pred_sx_straight = straighten_ski_points(ski_pred_sx)
        
        ski_gt_dx_straight = straighten_ski_points(ski_gt_dx)
        ski_gt_sx_straight = straighten_ski_points(ski_gt_sx)
        
        loss_dx = mse_loss(ski_pred_dx_straight, ski_gt_dx_straight)
        loss_sx = mse_loss(ski_pred_sx_straight, ski_gt_sx_straight)
        loss = (loss_dx + loss_sx) / 2.0

        self.log("train_loss", loss, prog_bar=True)
        return loss


    def validation_step(self, batch, batch_idx):
        inputs = batch['input_2d']
        x = inputs[:, :, :2].clone()
        target = x.clone()

        reconstructed, mask_idx = self(x, self.FLAGS)

        mse_loss = nn.MSELoss()
        
        ski_indices_dx = torch.tensor([19, 20, 21, 22], device=reconstructed.device).long()
        ski_indices_sx = torch.tensor([12, 13, 14, 15], device=reconstructed.device).long()
        
        ski_pred_dx = reconstructed[:, ski_indices_dx, :]
        ski_pred_sx = reconstructed[:, ski_indices_sx, :]
        ski_gt_dx = target[:, ski_indices_dx, :]
        ski_gt_sx = target[:, ski_indices_sx, :]
        
        ski_pred_dx_straight = straighten_ski_points(ski_pred_dx)
        ski_pred_sx_straight = straighten_ski_points(ski_pred_sx)
        ski_gt_dx_straight = straighten_ski_points(ski_gt_dx)
        ski_gt_sx_straight = straighten_ski_points(ski_gt_sx)
        
        loss_dx = mse_loss(ski_pred_dx_straight, ski_gt_dx_straight)
        loss_sx = mse_loss(ski_pred_sx_straight, ski_gt_sx_straight)
        loss = (loss_dx + loss_sx) / 2.0
            
        self.log("val_loss", loss, prog_bar=True)
        return loss


    def test_step(self, batch, batch_idx):
        inputs = batch['input_2d']
        x = inputs[:, :, :2].clone()
        target = x.clone()
        
        reconstructed, _ = self(x, self.FLAGS)

        if batch_idx == 0:
            print("\n=== DEBUG FIRST TEST BATCH ===")
            print(f"Ankle 11 GT: {target[0, 11, :]}")
            print(f"Ski point 12 (Left tip) GT: {target[0, 12, :]}")
            print(f"Ski point 12 (Left tip) PRED: {reconstructed[0, 12, :]}")
            print(f"Distance GT 11-12: {torch.norm(target[0, 11, :] - target[0, 12, :]):.4f}")
            print(f"Distance PRED 11-12: {torch.norm(target[0, 11, :] - reconstructed[0, 12, :]):.4f}")
            print(f"\nAnkle 18 GT: {target[0, 18, :]}")
            print(f"Ski point 19 (Right tip) GT: {target[0, 19, :]}")
            print(f"Ski point 19 (Right tip) PRED: {reconstructed[0, 19, :]}")
            print(f"Distance GT 18-19: {torch.norm(target[0, 18, :] - target[0, 19, :]):.4f}")
            print(f"Distance PRED 18-19: {torch.norm(target[0, 18, :] - reconstructed[0, 19, :]):.4f}")
            print("================================\n")
        
        total_loss = nn.MSELoss()(reconstructed, target)
        
        ski_joints = [19, 20, 21, 22, 12, 13, 14, 15]
        ski_indices = torch.tensor(ski_joints, device=reconstructed.device).long()
        ski_loss = nn.MSELoss()(reconstructed[:, ski_indices, :], target[:, ski_indices, :])
       
        skier_joints = [i for i in range(23) if i not in ski_joints]
        skier_indices = torch.tensor(skier_joints, device=reconstructed.device).long()
        skier_loss = nn.MSELoss()(reconstructed[:, skier_indices, :], target[:, skier_indices, :])
       
        self.log("test_loss_total", total_loss, on_step=True, on_epoch=True, logger=True)
        self.log("test_loss_ski", ski_loss, on_step=True, on_epoch=True, logger=True)
        self.log("test_loss_skier", skier_loss, on_step=True, on_epoch=True, logger=True)
      
        reconstructed_straight = reconstructed.clone()
        
        ski_indices_dx = torch.tensor([19, 20, 21, 22], device=reconstructed.device).long()
        ski_dx = reconstructed[:, ski_indices_dx, :]
        ski_dx_straight = straighten_ski_points(ski_dx)
        reconstructed_straight[:, ski_indices_dx, :] = ski_dx_straight
        
        ski_indices_sx = torch.tensor([12, 13, 14, 15], device=reconstructed.device).long()
        ski_sx = reconstructed[:, ski_indices_sx, :]
        ski_sx_straight = straighten_ski_points(ski_sx)
        reconstructed_straight[:, ski_indices_sx, :] = ski_sx_straight
        
        self.test_preds.append(reconstructed_straight.detach().cpu())
        self.test_gts.append(target.detach().cpu()) 
        
        return ski_loss

    def on_test_epoch_end(self):
        """
        This function is called ONCE at the end of the test.
        Here we save everything to file.
        """
        
        if len(self.test_preds) > 0:
            all_preds = torch.cat(self.test_preds, dim=0).float().numpy()
            all_gts = torch.cat(self.test_gts, dim=0).float().numpy()
            
            save_path = "test_results_full.pkl"
            
            with open(save_path, 'wb') as f:
                pkl.dump({'pred': all_preds, 'gt': all_gts}, f)
                
            print(f"File saved: {save_path}")
            print(f"Total images: {all_preds.shape[0]}")
        else:
            print("⚠️ No prediction saved!")
        
        self.test_preds = []
        self.test_gts = []

    def configure_optimizers(self):
        lr = self.FLAGS.lr

        opt_encoder = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        scheduler_encoder = torch.optim.lr_scheduler.ExponentialLR(optimizer=opt_encoder, gamma=0.95)

        return [opt_encoder], [scheduler_encoder]