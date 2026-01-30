from distutils.command.config import config
#from datamodule_smpl import SMPLDataModule
#from datamodule_h36m import H36MDatamodule
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

FLAGS = flags.FLAGS

# wandb.require("service")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "0"

def init_all():
    warnings.filterwarnings("ignore")

    # enable cudnn and its inbuilt auto-tuner to find the best algorithm to use for your hardware
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # useful for run-time
    torch.backends.cudnn.deterministic = True

    # pl.seed_everything(FLAGS.seed)
    torch.cuda.empty_cache()

# def sweep_iteration():
    

def main(argv):
    init_all()
    wandb.init(project=FLAGS.project_name, name="From %s to %s" %(FLAGS.dataset,FLAGS.train_dataset))
    wandb_logger = WandbLogger()
    # config = wandb.config
    if FLAGS.mode == "train":
        #if FLAGS.dataset == 'SKIJUMP':
        dm = SkijumpDataModule(FLAGS)
        # elif FLAGS.dataset == 'PANOPTIC2':
        #     dm = PanopticDataModule(FLAGS)
        # elif FLAGS.dataset == 'CMU_all':
        #     dm = SMPLDataModule(FLAGS)
        model = AdaptationNetwork(FLAGS)
        # directory = "/home/giuliamartinelli/Documents/Code/UnsupervisedHMR/DomainAdaptationModule/Models/%s/%s" %(FLAGS.dataset,FLAGS.masking_mode)
        # if not os.path.exists(directory):
        #     os.makedirs(directory)
        trainer = Trainer(
            default_root_dir="", #directory, # quella che vogliamo tipo ""
            accelerator="auto",
            devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
            max_epochs=FLAGS.n_epochs,
            precision="bf16",
            gradient_clip_val=5,
            detect_anomaly=True,
            callbacks=[TQDMProgressBar(refresh_rate=20)],logger=wandb_logger)
        trainer.fit(model, dm)

    if FLAGS.mode == "demo":
        model = AdaptationNetwork(FLAGS)
    
        dm = SkijumpDataModule(FLAGS)
        # elif FLAGS.dataset == 'PANOPTIC2':
        #     dm = PanopticDataModule(FLAGS)
        # elif FLAGS.dataset == 'CMU_all':
        #     dm = SMPLDataModule(FLAGS)
        trainer = Trainer(
            # gpus=1,
            accelerator="auto",
            devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
            max_epochs=FLAGS.n_epochs,
            precision="bf16",
            gradient_clip_val=5,
            # track_grad_norm=2,
            detect_anomaly=True,
            callbacks=[TQDMProgressBar(refresh_rate=20)],logger=wandb_logger)
        trainer.test(model = model,dataloaders = dm,ckpt_path=FLAGS.load_checkpoint)
    

if __name__ == '__main__':
    app.run(main)