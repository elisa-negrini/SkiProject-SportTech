from absl import flags
import os
FLAGS = flags.FLAGS
# DIRECTORIES
flags.DEFINE_string('dataset_dir', "./dataset_preprocessed", 'Preprocessed dataset directory.')
flags.DEFINE_string('checkpoint_dir', './checkpoints', 'Base directory for saving checkpoints')
flags.DEFINE_string('dataset', "SKIJUMP", 'dataset name')
flags.DEFINE_string('load_checkpoint','','Checkpoint path')
flags.DEFINE_string('raw_dataset_dir', "./dataset", 'Raw dataset directory for preprocessing')
# Training Parameters
flags.DEFINE_float('lr',2e-4, 'Learning Rate')
flags.DEFINE_integer('num_workers', int(os.cpu_count() / 2), 'Number of workers.')
flags.DEFINE_integer('batch_size', 64, 'Batch size.')
flags.DEFINE_integer('n_epochs', 25, 'Number of training epochs.')
flags.DEFINE_list('ski_joints', [19,20,21,22,12,13,14,15], 'Indices of ski joints to predict')
#Mode
flags.DEFINE_string('mode','demo','Train Mode')
flags.DEFINE_string('project_name', 'SkiPose', 'Project name for logging.')
flags.DEFINE_integer('seed', 42, 'Random seed for reproducibility.')
# Preprocessing & Dataset
flags.DEFINE_integer('n_joints', 23, 'Number of target joints. 24,25,17')
flags.DEFINE_float('train_split', 0.7, 'Training data split ratio')
flags.DEFINE_float('val_split', 0.85, 'Validation split ratio (cumulative)')
