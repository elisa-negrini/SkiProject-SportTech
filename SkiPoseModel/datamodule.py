import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import os
import re
import cv2
from random import shuffle
import sys
import pickle as pkl

from utils import *
from torch.utils.data import DataLoader, Dataset
sys.path.append("..")
try:
    from utils import * 
except ImportError:
    pass

import pytorch_lightning as pl


class SkijumpDataModule(pl.LightningDataModule): # cambia nome in SkijumpDataModule
    
    def __init__(self, FLAGS):
        
        super().__init__()
        self.FLAGS = FLAGS
        self.batch_size = getattr(FLAGS, 'batch_size', 64)
        self.num_workers = getattr(FLAGS, 'num_workers', 4)

     # quetsa classe deve rimanere identica, possiamo modicare le cose dentro alle funioni ma non cambiarne i nomi
    def train_dataloader(self):
        working_dir = os.path.join(self.FLAGS.dataset_dir, self.FLAGS.dataset) ## da sistemare
        dataset_path = os.path.join(working_dir, 'train.pkl') ##

        dataset = SkijumpDataset(dataset_path, self.FLAGS)

        train_loader = DataLoader(dataset, 
                                  shuffle=True, 
                                  pin_memory=True,
                                  num_workers=self.num_workers, 
                                  batch_size=self.batch_size,
                                  drop_last=True)

        return train_loader

    def val_dataloader(self):
        working_dir = os.path.join(self.FLAGS.dataset_dir, self.FLAGS.dataset)
        dataset_path = os.path.join(working_dir, 'test.pkl')

        dataset = SkijumpDataset(dataset_path, self.FLAGS)

        val_loader = DataLoader(dataset, 
                                shuffle=False, 
                                pin_memory=True,
                                num_workers=self.num_workers, 
                                batch_size=self.batch_size, 
                                drop_last=True)

        return val_loader

    def test_dataloader(self):
        working_dir = os.path.join(self.FLAGS.dataset_dir, self.FLAGS.dataset)
        dataset_path = os.path.join(working_dir, 'test.pkl')

        dataset = SkijumpDataset(dataset_path, self.FLAGS)

        test_loader = DataLoader(dataset, 
                                 shuffle=False, 
                                 pin_memory=True,
                                 num_workers=self.num_workers, 
                                 batch_size=self.batch_size, 
                                 drop_last=True)

        return test_loader


class SkijumpDataset(Dataset): # cambia nome in SkijumpDataset
    ''' In:
            data_path (string): path to the dataset split folder, i.e. train/valid/test
            transform (callable, optional): transform to be applied on a sample.
        Out:
            sample (dict): sample data and respective label'''


    def __init__(self, data_path, FLAGS=None):
        """
        Args:
            data_numpy (np.array): The 2D skeleton data loaded from pkl. Shape (N, 23, 3)
            flags: Optional flags/config argument (kept for compatibility).
        """
        self.flags = FLAGS
        self.data_path = data_path
        
        # --- CARICAMENTO DATI CORRETTO ---
        if os.path.exists(data_path):
            with open(data_path, 'rb') as f:
                loaded_data = pkl.load(f)
            
            # Controllo se è un dizionario (come nel tuo preprocess.py)
            if isinstance(loaded_data, dict):
                # Cerchiamo la chiave giusta. Nel tuo preprocess usi 'openpose_2d'
                if 'openpose_2d' in loaded_data:
                    self.data = loaded_data['openpose_2d']
                elif 'input_2d' in loaded_data: # fallback
                    self.data = loaded_data['input_2d']
                else:
                    # Se non trova le chiavi, prende la prima disponibile (rischio!)
                    keys = list(loaded_data.keys())
                    print(f"Warning: Key 'openpose_2d' not found. Using '{keys[0]}'")
                    self.data = loaded_data[keys[0]]
            else:
                # Se è già una lista o array
                self.data = loaded_data
        else:
            raise FileNotFoundError(f"File pickle non trovato: {data_path}")

    def __getitem__(self, index):
        # 1. Get the pose (already normalized and centered as per user instruction)
        # Shape: (23, 3) -> [x, y, confidence]
        pose = self.data[index]
        
        # 2. Convert to PyTorch Tensor (Float32)
        pose_tensor = torch.from_numpy(pose).float()
        
        # 3. Handle Input Dimensions
        # Usually networks expect (x, y). If your network needs confidence, keep all 3.
        # Assuming we pass everything for now.
        input_data = pose_tensor  # (23, 3)
        
        # 4. Prepare Output Dictionary
        # We use the same keys as typical domain adaptation codes to avoid changing model.py too much
        return {
            'input_2d': input_data,
            'target_2d': input_data, # For reconstruction/autoencoder task, target is input
             # Adding a dummy 'root' or 'target_3d' might be needed ONLY if model.py crashes asking for it.
             # For now, we keep it clean.
        }
    
    def __len__(self):
        return len(self.data)