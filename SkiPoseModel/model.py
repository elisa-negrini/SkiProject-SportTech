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
matplotlib.use('Agg') # Assicurati che ci sia!
import matplotlib.pyplot as plt


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
        """
        z shape: (Batch, Joints, 2)
        """
        mask_indices = []

        if FLAGS.mode == 'train':
            # CORREZIONE: Indici 0-based per gli sci
            ski_joints = [19, 20, 21, 22, 12, 13, 14, 15]  # <--- CORRETTO!
            mask_indices = torch.tensor(ski_joints, device=z.device).long()
            z[:, mask_indices, :] = 0.0
        
        elif FLAGS.mode == 'demo':
            ski_joints = [19, 20, 21, 22, 12, 13, 14, 15]  # <--- CORRETTO!
            mask_indices = torch.tensor(ski_joints, device=z.device).long()
            z[:, mask_indices, :] = 0.0

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
        target = x.clone()  # Target contiene le posizioni reali degli sci

        reconstructed, mask_idx = self(x, self.FLAGS)

        mse_loss = nn.MSELoss()
        
        # Loss SOLO sugli sci mascherati
        ski_joints = [19, 20, 21, 22, 12, 13, 14, 15]  # <--- INDICI CORRETTI 0-based
        ski_indices = torch.tensor(ski_joints, device=reconstructed.device).long()
        
        loss = mse_loss(reconstructed[:, ski_indices, :], target[:, ski_indices, :])

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs = batch['input_2d']
        x = inputs[:, :, :2].clone()
        target = x.clone()

        reconstructed, mask_idx = self(x, self.FLAGS)

        mse_loss = nn.MSELoss()
        
        # Loss SOLO sugli sci
        ski_joints = [19, 20, 21, 22, 12, 13, 14, 15]  # <--- INDICI CORRETTI 0-based
        ski_indices = torch.tensor(ski_joints, device=reconstructed.device).long()
        
        loss = mse_loss(reconstructed[:, ski_indices, :], target[:, ski_indices, :])
            
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
                
        inputs = batch['input_2d']
        x = inputs[:, :, :2].clone()
        target = x.clone()
        
        reconstructed, _ = self(x, self.FLAGS)
        
        total_loss = nn.MSELoss()(reconstructed, target)
        self.log("test_loss_total", total_loss)

        # Loss SOLO sugli sci
        ski_joints = [19, 20, 21, 22, 12, 13, 14, 15]
        ski_indices = torch.tensor(ski_joints, device=reconstructed.device).long()
        ski_loss = nn.MSELoss()(reconstructed[:, ski_indices, :], target[:, ski_indices, :])
        self.log("test_loss_ski", ski_loss)
        
        # Loss SOLO sullo sciatore
        skier_joints = [i for i in range(23) if i not in ski_joints]
        skier_indices = torch.tensor(skier_joints, device=reconstructed.device).long()
        skier_loss = nn.MSELoss()(reconstructed[:, skier_indices, :], target[:, skier_indices, :])
        self.log("test_loss_skier", skier_loss)
        
        # --- VISUALIZZAZIONE SU WANDB CON CONNESSIONI E CONSTRAINT ---
        if batch_idx == 0:
            print("\n" + "="*60)
            print("🎨 INIZIO PLOTTING BATCH 0...")
            print("="*60)
            
            try:
                gt = target[0].float().cpu().numpy()
                pred_raw = reconstructed[0].float().cpu().numpy()
                
                print(f"✅ Dati caricati - GT shape: {gt.shape}, Pred shape: {pred_raw.shape}")
                
                # ========== APPLICA CONSTRAINT LINEARE AGLI SCI ==========
                pred = pred_raw.copy()
                
                # Sci Destro: joints [22, 19, 20, 21]
                ski_dx_indices = [22, 19, 20, 21]
                ski_dx_points = pred_raw[ski_dx_indices]
                coeffs_dx = np.polyfit(ski_dx_points[:, 0], ski_dx_points[:, 1], 1)
                pred[ski_dx_indices, 1] = np.polyval(coeffs_dx, pred[ski_dx_indices, 0])
                
                print(f"✅ Fit Sci Destro: y = {coeffs_dx[0]:.3f}x + {coeffs_dx[1]:.3f}")
                
                # Sci Sinistro: joints [15, 12, 13, 14]
                ski_sx_indices = [15, 12, 13, 14]
                ski_sx_points = pred_raw[ski_sx_indices]
                coeffs_sx = np.polyfit(ski_sx_points[:, 0], ski_sx_points[:, 1], 1)
                pred[ski_sx_indices, 1] = np.polyval(coeffs_sx, pred[ski_sx_indices, 0])
                
                print(f"✅ Fit Sci Sinistro: y = {coeffs_sx[0]:.3f}x + {coeffs_sx[1]:.3f}")
                
                # ========== DEFINISCI CONNESSIONI ==========
                skier_connections = [
                    [0, 1], [1, 2], [2, 3], [3, 4],
                    [1, 5], [5, 6], [6, 7],
                    [1, 8],
                    [8, 9], [9, 10], [10, 11],
                    [8, 16], [16, 17], [17, 18],
                ]
                
                ski_connections = [
                    [22, 19], [19, 20], [20, 21],
                    [15, 12], [12, 13], [13, 14],
                ]
                
                print("✅ Connessioni definite")
                
                # ========== PLOT ==========
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
                
                print("✅ Figure creata")
                
                # --- SUBPLOT 1: RAW PREDICTION ---
                ax1.set_title(f'RAW - Ski Loss: {ski_loss:.4f}', fontsize=14, fontweight='bold')
                
                skier_mask = [i for i in range(23) if i not in ski_joints]
                ax1.scatter(gt[skier_mask, 0], gt[skier_mask, 1], 
                        c='blue', s=120, label='Sciatore GT', marker='o', 
                        edgecolors='black', linewidths=2, zorder=5)
                ax1.scatter(gt[ski_joints, 0], gt[ski_joints, 1], 
                        c='green', s=120, label='Sci GT', marker='s', 
                        edgecolors='black', linewidths=2, zorder=5)
                ax1.scatter(pred_raw[skier_mask, 0], pred_raw[skier_mask, 1], 
                        c='cyan', s=60, label='Sciatore Pred', marker='o', 
                        alpha=0.7, zorder=4)
                ax1.scatter(pred_raw[ski_joints, 0], pred_raw[ski_joints, 1], 
                        c='red', s=120, label='Sci Pred (raw)', marker='x', 
                        linewidths=3, zorder=4)
                
                for conn in skier_connections:
                    ax1.plot([gt[conn[0], 0], gt[conn[1], 0]], 
                            [gt[conn[0], 1], gt[conn[1], 1]], 
                            'b-', linewidth=2, alpha=0.6, zorder=2)
                for conn in ski_connections:
                    ax1.plot([gt[conn[0], 0], gt[conn[1], 0]], 
                            [gt[conn[0], 1], gt[conn[1], 1]], 
                            'g-', linewidth=3, alpha=0.8, zorder=2)
                for conn in skier_connections:
                    ax1.plot([pred_raw[conn[0], 0], pred_raw[conn[1], 0]], 
                            [pred_raw[conn[0], 1], pred_raw[conn[1], 1]], 
                            'c--', linewidth=1.5, alpha=0.5, zorder=1)
                for conn in ski_connections:
                    ax1.plot([pred_raw[conn[0], 0], pred_raw[conn[1], 0]], 
                            [pred_raw[conn[0], 1], pred_raw[conn[1], 1]], 
                            'r--', linewidth=2, alpha=0.6, zorder=1)
                
                ax1.legend(loc='upper right', fontsize=10)
                ax1.invert_yaxis()
                ax1.grid(True, alpha=0.3)
                ax1.set_xlabel("X", fontsize=12)
                ax1.set_ylabel("Y", fontsize=12)
                
                print("✅ Subplot 1 (RAW) completato")
                
                # --- SUBPLOT 2: WITH CONSTRAINT ---
                ax2.set_title(f'WITH CONSTRAINT - Ski Loss: {ski_loss:.4f}', fontsize=14, fontweight='bold')
                
                ax2.scatter(gt[skier_mask, 0], gt[skier_mask, 1], 
                        c='blue', s=120, label='Sciatore GT', marker='o', 
                        edgecolors='black', linewidths=2, zorder=5)
                ax2.scatter(gt[ski_joints, 0], gt[ski_joints, 1], 
                        c='green', s=120, label='Sci GT', marker='s', 
                        edgecolors='black', linewidths=2, zorder=5)
                ax2.scatter(pred[skier_mask, 0], pred[skier_mask, 1], 
                        c='cyan', s=60, label='Sciatore Pred', marker='o', 
                        alpha=0.7, zorder=4)
                ax2.scatter(pred[ski_joints, 0], pred[ski_joints, 1], 
                        c='red', s=120, label='Sci Pred (constrained)', marker='x', 
                        linewidths=3, zorder=4)
                
                for conn in skier_connections:
                    ax2.plot([gt[conn[0], 0], gt[conn[1], 0]], 
                            [gt[conn[0], 1], gt[conn[1], 1]], 
                            'b-', linewidth=2, alpha=0.6, zorder=2)
                for conn in ski_connections:
                    ax2.plot([gt[conn[0], 0], gt[conn[1], 0]], 
                            [gt[conn[0], 1], gt[conn[1], 1]], 
                            'g-', linewidth=3, alpha=0.8, zorder=2)
                for conn in skier_connections:
                    ax2.plot([pred[conn[0], 0], pred[conn[1], 0]], 
                            [pred[conn[0], 1], pred[conn[1], 1]], 
                            'c--', linewidth=1.5, alpha=0.5, zorder=1)
                for conn in ski_connections:
                    ax2.plot([pred[conn[0], 0], pred[conn[1], 0]], 
                            [pred[conn[0], 1], pred[conn[1], 1]], 
                            'r--', linewidth=2, alpha=0.6, zorder=1)
                
                # Linee di fit
                x_range = np.linspace(pred[ski_dx_indices, 0].min() - 0.05, 
                                    pred[ski_dx_indices, 0].max() + 0.05, 100)
                ax2.plot(x_range, np.polyval(coeffs_dx, x_range), 
                        'r:', linewidth=2, alpha=0.5, label='Fit Sci Destro')
                
                x_range_sx = np.linspace(pred[ski_sx_indices, 0].min() - 0.05, 
                                        pred[ski_sx_indices, 0].max() + 0.05, 100)
                ax2.plot(x_range_sx, np.polyval(coeffs_sx, x_range_sx), 
                        'r:', linewidth=2, alpha=0.5, label='Fit Sci Sinistro')
                
                ax2.legend(loc='upper right', fontsize=10)
                ax2.invert_yaxis()
                ax2.grid(True, alpha=0.3)
                ax2.set_xlabel("X", fontsize=12)
                ax2.set_ylabel("Y", fontsize=12)
                
                print("✅ Subplot 2 (CONSTRAINED) completato")
                
                plt.tight_layout()
                
                # Salva localmente per verifica
                output_file = 'test_result_batch_0.png'
                fig.savefig(output_file, dpi=150, bbox_inches='tight')
                print(f"✅ Plot salvato localmente: {output_file}")
                
                # Invia a WandB
                try:
                    wandb.log({"test/confronto_scheletri": wandb.Image(fig)})
                    print("✅ Plot inviato a WandB")
                except Exception as wandb_error:
                    print(f"⚠️ Errore invio WandB: {wandb_error}")
                
                plt.close(fig)
                print("✅ Figure chiusa")
                print("="*60)
                print("✅ PLOTTING COMPLETATO CON SUCCESSO!")
                print("="*60 + "\n")
                
            except Exception as e:
                print(f"❌ ERRORE NEL PLOT: {e}")
                import traceback
                traceback.print_exc()
                print("="*60 + "\n")
        
        return ski_loss

    def configure_optimizers(self):
        lr = self.FLAGS.lr

        opt_encoder = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        scheduler_encoder = torch.optim.lr_scheduler.ExponentialLR(optimizer=opt_encoder, gamma=0.95)

        return [opt_encoder], [scheduler_encoder]