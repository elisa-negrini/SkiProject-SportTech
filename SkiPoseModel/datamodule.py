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


class SkijumpDataModule(pl.LightningDataModule):
    
    def __init__(self, FLAGS):
        
        super().__init__()
        self.FLAGS = FLAGS
        self.batch_size = getattr(FLAGS, 'batch_size', 64)
        self.num_workers = getattr(FLAGS, 'num_workers', 2)

    def train_dataloader(self):
        working_dir = os.path.join(self.FLAGS.dataset_dir, self.FLAGS.dataset)
        dataset_path = os.path.join(working_dir, 'train.pkl')

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


class SkijumpDataset(Dataset):
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
        
        if os.path.exists(data_path):
            with open(data_path, 'rb') as f:
                loaded_data = pkl.load(f)
            
            if isinstance(loaded_data, dict):
                if 'openpose_2d' in loaded_data:
                    self.data = loaded_data['openpose_2d']
                elif 'input_2d' in loaded_data: 
                    self.data = loaded_data['input_2d']
                else:
                    keys = list(loaded_data.keys())
                    print(f"Warning: Key 'openpose_2d' not found. Using '{keys[0]}'")
                    self.data = loaded_data[keys[0]]
            else:
                self.data = loaded_data
        else:
            raise FileNotFoundError(f"File pickle not found: {data_path}")

    def __getitem__(self, index):
        # 1. Get the pose 
        pose = self.data[index]
        
        # 2. Convert to PyTorch Tensor (Float32)
        pose_tensor = torch.from_numpy(pose).float()
        
        # 3. Handle Input Dimensions
        input_data = pose_tensor[:, :2]
        
        # 4. Prepare Output Dictionary
        return {
            'input_2d': input_data,
            'target_2d': input_data,
        }
    
    def __len__(self):
        return len(self.data)