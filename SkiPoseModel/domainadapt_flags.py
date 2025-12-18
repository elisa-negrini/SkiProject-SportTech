from absl import flags
import os
FLAGS = flags.FLAGS
# DIRECTORIES
flags.DEFINE_string('dataset_dir', "/home/sporttechstudents/Documents/Dataset/SkiPose", 'Dataset directory.')
flags.DEFINE_string('dataset', "SKIJUMP", 'Dataset name. Choose btw: CMU_all, PANOPTIC2, H36M')
flags.DEFINE_string('load_checkpoint','/home/sporttechstudents/Documents/checkpoint/SMPL.ckpt','Checkpoint path')
# input-output
flags.DEFINE_string('train_dataset','SKIJUMP','Choose train dataset for testing. H36M, CMU_all, PANOPTIC2')
flags.DEFINE_integer('n_joints', 23, 'Number of target joints. 24,25,17')
# Training Parameters
flags.DEFINE_float('lr',2e-4, 'Learning Rate')
flags.DEFINE_float('b1', 0.5, 'Beta1')
flags.DEFINE_float('b2', 0.999, 'Beta2')
flags.DEFINE_integer('num_workers', int(os.cpu_count() / 2), 'Number of workers.')
flags.DEFINE_integer('batch_size', 64, 'Batch size.')
flags.DEFINE_integer('n_epochs', 25, 'Number of training epochs.')
flags.DEFINE_integer('masked_joints',6,'Number of masked joints: 13,14,6')
flags.DEFINE_string('masking_mode','end_effectors','Choose masking mode. random, noise, end-effectors')

#Mode
flags.DEFINE_string('mode','demo','Train Mode')
flags.DEFINE_bool('rotate',False,'Plot test different view')

## added
flags.DEFINE_string('project_name', 'SkiPose', 'Project name for logging.')
flags.DEFINE_integer('seed', 42, 'Random seed for reproducibility.')
