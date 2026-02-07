from datamodule import SkijumpDataModule
from model import AdaptationNetwork
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import os
import wandb
import numpy as np
import domainadapt_flags
from absl import app
from absl import flags
import warnings
from datetime import datetime
from pytorch_lightning.callbacks import ModelCheckpoint

FLAGS = flags.FLAGS

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "0"

def init_all():
    warnings.filterwarnings("ignore")

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    torch.backends.cudnn.deterministic = True

    torch.cuda.empty_cache()
   

def main(argv):
    init_all()

    wandb_logger = WandbLogger(
        project=FLAGS.project_name, 
        name=f"TEST_{FLAGS.dataset}", 
        log_model=False
    )

    if FLAGS.mode == "train":
        dm = SkijumpDataModule(FLAGS)
        model = AdaptationNetwork(FLAGS)
        run_name = f"{FLAGS.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        checkpoint_path = os.path.join(FLAGS.checkpoint_dir, run_name)
        os.makedirs(checkpoint_path, exist_ok=True)
        trainer = Trainer(
            default_root_dir=checkpoint_path,
            accelerator="auto",
            devices=1 if torch.cuda.is_available() else None,
            max_epochs=FLAGS.n_epochs,
            precision=32, #"bf16",
            gradient_clip_val=5,
            detect_anomaly=True,
            callbacks=[
                TQDMProgressBar(refresh_rate=20), ModelCheckpoint(
                                                    dirpath=checkpoint_path,
                                                    filename='{epoch:02d}-{val_loss:.4f}',
                                                    save_top_k=3,
                                                    save_last=True,
                                                    monitor='val_loss',
                                                    mode='min'
                                                )
                        ],
            logger=wandb_logger)
        trainer.fit(model, dm)

    if FLAGS.mode == "demo" or FLAGS.mode == "test":
        model = AdaptationNetwork(FLAGS)
        dm = SkijumpDataModule(FLAGS)
        trainer = Trainer(
            # gpus=1,
            accelerator="auto",
            devices=1 if torch.cuda.is_available() else None,
            max_epochs=FLAGS.n_epochs,
            precision=32, #"bf16-mixed",
            gradient_clip_val=5,
            # track_grad_norm=2,
            detect_anomaly=True,
            callbacks=[TQDMProgressBar(refresh_rate=20)],
            logger=wandb_logger)
        trainer.test(model = model,dataloaders = dm,ckpt_path=FLAGS.load_checkpoint)

        wandb.finish()
    

if __name__ == '__main__':
    app.run(main)