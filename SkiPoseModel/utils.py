from cProfile import label
from multiprocessing import connection
import torch
import numpy as np
from torch.utils.data import Dataset
import wandb
import os, glob
import re
import cv2
from random import shuffle
import matplotlib.pyplot as plt
from einops import rearrange
from matplotlib.gridspec import GridSpec
from pydoc import classname
import torch.nn as nn

def connect3d(x,y,z,p1,p2):
    x1, x2 = x[p1], x[p2]
    y1, y2 = y[p1], y[p2]
    z1,z2 = z[p1],z[p2]
    return x1,x2,y1,y2,z1,z2

def x_rotation(theta):
    """Rotates 3-D vector around x-axis"""
    R = np.array([[1,0,0],[0,np.cos(theta),-np.sin(theta)],[0, np.sin(theta), np.cos(theta)]],dtype=np.float)
    return R

def y_rotation(theta):
    """Rotates 3-D vector around y-axis"""
    R = np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta), 0, np.cos(theta)]],dtype=np.float)
    return R

def z_rotation(theta):
    """Rotates 3-D vector around z-axis"""
    R = np.array([[np.cos(theta), -np.sin(theta),0],[np.sin(theta), np.cos(theta),0],[0,0,1]],dtype=np.float)
    return R

def rotate_2d(pt_2d, rot_rad):
    x = pt_2d[0]
    y = pt_2d[1]
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    xx = x * cs - y * sn
    yy = x * sn + y * cs
    return np.array([xx, yy], dtype=np.float32)

def translation(v):
    v_new = np.empty(shape=v.shape)
    v_new[:,0,:] = np.array([0,0,0])
    for i in range(1,v.shape[1]):
        v_new[:,i,0] = np.array(v[:,i,0] -v[:,0,0])
        v_new[:,i,1] = np.array(v[:,i,1] - v[:,0,1])
        v_new[:,i,2] = np.array(v[:,i,2] - v[:,0,2])
    return v_new

def plot_smpl(input):
    plt.style.use('ggplot')
    img = {}
    fig = plt.figure(figsize=(5,5),dpi=300)
    ax = fig.add_subplot(1,1,1,projection='3d')

    input_labels = ['pelvis','left_hip','right_hip','spine1','left_knee','right_knee','spine2','left_ankle','right_ankle',
                    'spine3','left_foot','right_foot','neck','left_collar','right_collar','head','left_shoulder','right_shoulder',
                    'left_elbow','right_elbow','left_wrist','right_wrist','left_hand','right_hand']
    input_connections = [[0,1],[0,2],[0,3],[1,4],[2,5],[3,6],[4,7],[5,8],[6,9],[7,10],[8,11],[9,12],[9,13],[9,14],[12,15],[13,16],[14,17],
                    [16,18],[17,19],[18,20],[19,21],[20,22],[21,23]]
    color_bones = '-b'
    color_joints = 'b'

    for i,connections in enumerate(input_connections):
        x1,x2,y1,y2,z1,z2 = connect3d(input[:,0], input[:,1],input[:,2],connections[0],connections[1])
        ax.plot([x1,x2],[y1,y2],[z1,z2],color_bones)
    for i,l in enumerate(input_labels):
        ax.scatter(input[i,0], input[i,1],input[i,2],color=color_joints)


    ax.set_title('Domain Adaptation',fontsize=30)
    ax.set_xlim([-0.1, 0.1])
    ax.set_ylim([-0.1, 0.1])
    ax.set_zlim([-0.1, 0.1])
    ax.w_xaxis.pane.fill = False
    ax.w_yaxis.pane.fill = False
    ax.w_zaxis.pane.fill = False
    ax.set_xticks([]) 
    ax.set_yticks([]) 
    ax.set_zticks([])
    ax.set_axis_off()

    plt.show()

def plot_panoptic(input):
    plt.style.use('ggplot')
    img = {}
    fig = plt.figure(figsize=(5,5),dpi=300)
    ax = fig.add_subplot(1,1,1,projection='3d')

    input_labels = ['pelvis',"neck","nose",'right_shoulder','right_elbow','right_wrist','right_hip','right_knee','right_ankle','left_shoulder','left_elbow','left_wrist',
                    'left_hip','left_knee','left_ankle','right_eye','right_ear','left_eye','left_ear','right_b_toe','right_s_toe','right_hill','left_b_toe','left_s_toe','left_hill']
    input_connections = [[0,1],[1,2],[1,3],[3,4],[4,5],[0,6],[6,7],[7,8],[1,9],[9,10],[10,11],[0,12],[12,13],[13,14],[2,15],[15,16],[2,17],[17,18],[8,19],[19,20],[8,21],
                    [14,22],[22,23],[14,24]]
    
    color_bones = '-b'
    color_joints = 'b'


    for i,connections in enumerate(input_connections):
        x1,x2,y1,y2,z1,z2 = connect3d(input[:,0], input[:,1],input[:,2],connections[0],connections[1])
        ax.plot([x1,x2],[y1,y2],[z1,z2],color_bones)
    for i,l in enumerate(input_labels):
        ax.scatter(input[i,0], input[i,1],input[i,2],color=color_joints)


    ax.set_title('Domain Adaptation',fontsize=30)
    ax.set_xlim([-0.1, 0.1])
    ax.set_ylim([-0.1, 0.1])
    ax.set_zlim([-0.1, 0.1])
    ax.w_xaxis.pane.fill = False
    ax.w_yaxis.pane.fill = False
    ax.w_zaxis.pane.fill = False
    ax.set_xticks([]) 
    ax.set_yticks([]) 
    ax.set_zticks([])
    ax.set_axis_off()

    plt.show()


    

def plot_h36m(input):
    plt.style.use('ggplot')
    img = {}
    fig = plt.figure(figsize=(5,5),dpi=300)
    ax = fig.add_subplot(1,1,1,projection='3d')

    labels = ["hip",'right_hip','right_knee','right_ankle','left_hip','left_knee','left_ankle','chest','neck','chin','head',
                    'right_shoulder','right_elbow','right_wrist','left_shoulder','left_elbow','left_wrist']
    labels_all = ["hip",'right_hip','right_knee','right_ankle', 'right_toe','left_hip','left_knee','left_ankle','left_toe','spine1','chest','spine2','neck','head',
                    'right_collar','right_shoulder','right_elbow','right_wrist', 'right_hand','left_collar','left_shoulder','left_elbow','left_wrist', 'left_hand']
    # labels_all = ["hip",'right_hip','right_knee','right_ankle','left_hip','left_knee','left_ankle','chest','neck','chin','head',
    #                 'right_shoulder','right_elbow','right_wrist', 'right_hand','left_shoulder','left_elbow','left_wrist', 'left_hand']
    connections = [[0,1],[1,2],[2,3],[0,4],[4,5],[5,6],[0,7],[7,8],[8,9],[9,10],[8,11],[11,12],
                    [12,13],[8,14],[14,15],[15,16]]
    connections_all = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],[9,10],[10,11],[11,12],[12,13],[11,14],[14,15],
                    [15,16],[16,17],[17,18],[11,19],[19,20],[20,21],[21,22]]
    # connections_all = [[0,1],[1,2],[2,3],[0,4],[4,5],[5,6],[0,7],[7,8],[8,9],[9,10],[8,11],[11,12],
    #                 [12,13],[13,14],[8,15],[15,16],[16,17],[17,18]]
    color_bones = '-r'
    color_joints = 'r'


    # R = x_rotation(np.radians(90))
    # input = np.dot(input,R)

    for i,connections in enumerate(connections_all):
        x1,x2,y1,y2,z1,z2 = connect3d(input[:,0], input[:,1],input[:,2],connections[0],connections[1])
        ax.plot([x1,x2],[y1,y2],[z1,z2],color_bones)
    for i,l in enumerate(labels_all):
        ax.scatter(input[i,0], input[i,1],input[i,2],color=color_joints)


    ax.set_title('Domain Adaptation',fontsize=30)
    ax.set_xlim([-0.1, 0.1])
    ax.set_ylim([-0.1, 0.1])
    ax.set_zlim([-0.1, 0.1])
    ax.w_xaxis.pane.fill = False
    ax.w_yaxis.pane.fill = False
    ax.w_zaxis.pane.fill = False
    ax.set_xticks([]) 
    ax.set_yticks([]) 
    ax.set_zticks([])
    ax.set_axis_off()

    plt.show()

def get_missing_joints(input,hand_offset,foot_offset,spine_offset):


    left_arm_direction = input[:,10,:] - input[:,11,:]
    right_arm_direction = input[:,4,:] - input[:,5,:]

    left_hand_position = input[:,11,:] + left_arm_direction * hand_offset # np.array([hand_offset, 0, 0])
    right_hand_position = input[:,5,:] + right_arm_direction * hand_offset #np.array([hand_offset, 0, 0])

    # Calculate the vector along the leg
    left_leg_vector = input[:,13,:] - input[:,14,:]

    right_leg_vector = input[:,7,:] - input[:,8,:]

    left_leg_perpend = np.cross(left_leg_vector, [1, 1, 1])
    right_leg_perpend = np.cross(right_leg_vector, [1, 1, 1])

    left_foot_position = input[:,14,:] + foot_offset * left_leg_perpend
    right_foot_position = input[:,8,:] + foot_offset * right_leg_perpend

    spine_direction = input[:,0,:] - input[:,1,:]

    head_smpl = input[:,1,:] + spine_direction * spine_offset

    spine1 = (input[:,0,:] + input[:,1,:]) / 2

    spine2 = (spine1 + input[:,1,:]) / 2

    spine3 = spine1 + (1/4) * (input[:,1,:] - spine1)

    left_hip = (input[:,12,:] + input[:,13,:]) / 3
    right_hip = (input[:,6,:] + input[:,7,:]) / 3

    left_collar = (input[:,9,:] + spine3) / 2
    right_collar = (input[:,3,:] + spine3) / 2


    return left_hip, right_hip, left_hand_position, right_hand_position, left_foot_position, right_foot_position, spine1, spine2, spine3, head_smpl, left_collar, right_collar


def plot_testing(input, target,input_all,FLAGS,idx,show = False):
    plt.style.use('ggplot')
    img = {}
    fig = plt.figure(figsize=(40,30),dpi=300)
    # fig = plt.figure(figsize=(15,5),dpi=300)
    ax1= fig.add_subplot(1,4,1,projection='3d')
    ax= fig.add_subplot(1,4,2,projection='3d')
    ax2= fig.add_subplot(1,4,3,projection='3d')
    ax3= fig.add_subplot(1,4,4,projection='3d')
    

    if FLAGS.train_dataset == 'H36M' and FLAGS.dataset == 'CMU_all':
        
        target_labels = ["hip",'right_hip','right_knee','right_ankle','left_hip','left_knee','left_ankle','chest','neck','chin','head',
                    'right_shoulder','right_elbow','right_wrist','left_shoulder','left_elbow','left_wrist']
        target_connections = [[0,1],[1,2],[2,3],[0,4],[4,5],[5,6],[0,7],[7,8],[8,9],[9,10],[8,11],[11,12],
                        [12,13],[8,14],[14,15],[15,16]]
        color_bones_target = '-r'
        color_joints_target = 'r'

        input_labels = ['pelvis','left_hip','right_hip','spine1','left_knee','right_knee','spine2','left_ankle','right_ankle',
                    'spine3','left_foot','right_foot','neck','left_collar','right_collar','head','left_shoulder','right_shoulder',
                    'left_elbow','right_elbow','left_wrist','right_wrist','left_hand','right_hand']
        input_connections = [[0,1],[0,2],[0,3],[1,4],[2,5],[3,6],[4,7],[5,8],[6,9],[7,10],[8,11],[9,12],[9,13],[9,14],[12,15],[13,16],[14,17],
                        [16,18],[17,19],[18,20],[19,21],[20,22],[21,23]]
        color_bones_input = '-g'
        color_joints_input = 'g'


        joints_keep = [0,4,5,6,7,8,12,16,17,18,19,20,21]
        joints_masked = [1,4,9,10]
        
        bone_map_input = [[0,1],[0,2],[1,4],[2,5],[9,13],[9,14],[13,16],[14,17],[7,10],[8,11],[21,23],[20,22],[0,3],[3,6],[9,12],[12,15]]
        bone_map_target = []
        
        for bone in target_connections:
            for index in joints_masked:
                if index in bone:
                    bone_map_target.append(bone)
                    break

    elif FLAGS.train_dataset == 'H36M' and FLAGS.dataset == 'PANOPTIC2':

        target_labels = ["hip",'right_hip','right_knee','right_ankle','left_hip','left_knee','left_ankle','chest','neck','chin','head',
                'right_shoulder','right_elbow','right_wrist','left_shoulder','left_elbow','left_wrist']
        target_connections = [[0,1],[1,2],[2,3],[0,4],[4,5],[5,6],[0,7],[7,8],[8,9],[9,10],[8,11],[11,12],
                        [12,13],[8,14],[14,15],[15,16]]
        color_bones_target = '-r'
        color_joints_target = 'r'

        input_labels = ['pelvis',"neck","nose",'right_shoulder','right_elbow','right_wrist','right_hip','right_knee','right_ankle','left_shoulder','left_elbow','left_wrist',
                    'left_hip','left_knee','left_ankle','right_eye','right_ear','left_eye','left_ear','right_b_toe','right_s_toe','right_hill','left_b_toe','left_s_toe','left_hill']
        input_connections = [[0,1],[1,2],[1,3],[3,4],[4,5],[0,6],[6,7],[7,8],[1,9],[9,10],[10,11],[0,12],[12,13],[13,14],[2,15],[15,16],[2,17],[17,18],[8,19],[19,20],[8,21],
                        [14,22],[22,23],[14,24]]
        color_bones_input = '-b'
        color_joints_input = 'b'

        joints_keep = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
        joints_masked = [7,10]

        bone_map_input = [[0,2],[1,15],[15,16],[1,17],[17,18],[14,24],[14,22],[22,23],[8,21],[8,20],[19,20]]
        bone_map_target = []
        
        for bone in target_connections:
            for index in joints_masked:
                if index in bone:
                    bone_map_target.append(bone)
                    break

    elif FLAGS.train_dataset == 'H36M' and FLAGS.dataset == 'H36M':

        target_labels = ["hip",'right_hip','right_knee','right_ankle','left_hip','left_knee','left_ankle','chest','neck','chin','head',
                'right_shoulder','right_elbow','right_wrist','left_shoulder','left_elbow','left_wrist']
        target_connections = [[0,1],[1,2],[2,3],[0,4],[4,5],[5,6],[0,7],[7,8],[8,9],[9,10],[8,11],[11,12],
                        [12,13],[8,14],[14,15],[15,16]]
        color_bones_target = '-r'
        color_joints_target = 'r'

        input_labels = ["hip",'right_hip','right_knee','right_ankle','left_hip','left_knee','left_ankle','chest','neck','chin','head',
                'right_shoulder','right_elbow','right_wrist','left_shoulder','left_elbow','left_wrist']
        input_connections = [[0,1],[1,2],[2,3],[0,4],[4,5],[5,6],[0,7],[7,8],[8,9],[9,10],[8,11],[11,12],
                        [12,13],[8,14],[14,15],[15,16]]
        color_bones_input = '-b'
        color_joints_input = 'b'

        joints_keep = [0,3,6,10,16,13]
        joints_masked = [1,2,4,5,7,8,9,11,12,14,15]

        bone_map_input = [[0,2],[1,15],[15,16],[1,17],[17,18],[14,24],[14,22],[22,23],[8,21],[8,20],[19,20]]
        bone_map_target = []
        
        for bone in target_connections:
            for index in joints_masked:
                if index in bone:
                    bone_map_target.append(bone)
                    break

    
    elif FLAGS.train_dataset == 'CMU_all' and FLAGS.dataset == 'H36M':

        target_labels = ['pelvis','left_hip','right_hip','spine1','left_knee','right_knee','spine2','left_ankle','right_ankle',
                    'spine3','left_foot','right_foot','neck','left_collar','right_collar','head','left_shoulder','right_shoulder',
                    'left_elbow','right_elbow','left_wrist','right_wrist','left_hand','right_hand']
        target_connections = [[0,1],[0,2],[0,3],[1,4],[2,5],[3,6],[4,7],[5,8],[6,9],[7,10],[8,11],[9,12],[9,13],[9,14],[12,15],[13,16],[14,17],
                        [16,18],[17,19],[18,20],[19,21],[20,22],[21,23]]
        color_bones_target = '-g'
        color_joints_target = 'g'

        input_labels = ["hip",'right_hip','right_knee','right_ankle','left_hip','left_knee','left_ankle','chest','neck','chin','head',
                        'right_shoulder','right_elbow','right_wrist','left_shoulder','left_elbow','left_wrist']
        input_connections = [[0,1],[1,2],[2,3],[0,4],[4,5],[5,6],[0,7],[7,8],[8,9],[9,10],[8,11],[11,12],
                        [12,13],[8,14],[14,15],[15,16]]
        color_bones_input = '-r'
        color_joints_input = 'r'

        labels_all = ["hip",'right_hip','right_knee','right_ankle', 'right_toe','left_hip','left_knee','left_ankle','left_toe','chest','neck','chin','head',
                    'right_shoulder','right_elbow','right_wrist', 'right_hand','left_shoulder','left_elbow','left_wrist', 'left_hand']
        connections_all = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],[9,10],[10,11],[11,12],[10,13],[13,14],
                        [14,15],[15,16],[10,17],[17,18],[18,19],[19,20]]

        joints_keep = [0,2,3,5,6,7,8,11,12,13,14,15,16]
        joints_masked = [1,2,3,9,10,11,13,14,15,22,23]

        bone_map_input = [[0,1],[0,4],[8,9],[9,10],[4,5],[1,2]]
        bone_map_target = []
        
        for bone in target_connections:
            for index in joints_masked:
                if index in bone:
                    bone_map_target.append(bone)
                    break

    elif FLAGS.train_dataset == 'CMU_all' and FLAGS.dataset == 'PANOPTIC2':
    
        target_labels = ['pelvis','left_hip','right_hip','spine1','left_knee','right_knee','spine2','left_ankle','right_ankle',
                    'spine3','left_foot','right_foot','neck','left_collar','right_collar','head','left_shoulder','right_shoulder',
                    'left_elbow','right_elbow','left_wrist','right_wrist','left_hand','right_hand']
        target_connections = [[0,1],[0,2],[0,3],[1,4],[2,5],[3,6],[4,7],[5,8],[6,9],[7,10],[8,11],[9,12],[9,13],[9,14],[12,15],[13,16],[14,17],
                        [16,18],[17,19],[18,20],[19,21],[20,22],[21,23]]
        color_bones_target = '-g'
        color_joints_target = 'g'

        input_labels = ['pelvis',"neck","nose",'right_shoulder','right_elbow','right_wrist','right_hip','right_knee','right_ankle','left_shoulder','left_elbow','left_wrist',
                    'left_hip','left_knee','left_ankle','right_eye','right_ear','left_eye','left_ear','right_b_toe','right_s_toe','right_hill','left_b_toe','left_s_toe','left_hill']
        input_connections = [[0,1],[1,2],[1,3],[3,4],[4,5],[0,6],[6,7],[7,8],[1,9],[9,10],[10,11],[0,12],[12,13],[13,14],[2,15],[15,16],[2,17],[17,18],[8,19],[19,20],[8,21],
                        [14,22],[22,23],[14,24]]
        color_bones_input = '-b'
        color_joints_input = 'b'

        joints_keep = [13,14,22,6,7,8,19,3,4,5,9,10,11,0,1]
        joints_masked = [1,2,3,6,9,13,14,15,22,23]

        bone_map_input = [[1,2],[1,15],[15,16],[1,17],[17,18],[14,24],[14,22],[22,23],[8,21],[8,20],[19,20]]
        bone_map_target = []
        
        for bone in target_connections:
            for index in joints_masked:
                if index in bone:
                    bone_map_target.append(bone)
                    break

    elif FLAGS.train_dataset == 'PANOPTIC2' and FLAGS.dataset == 'H36M':
        
        target_labels = ['pelvis',"neck","nose",'right_shoulder','right_elbow','right_wrist','right_hip','right_knee','right_ankle','left_shoulder','left_elbow','left_wrist',
                    'left_hip','left_knee','left_ankle','right_eye','right_ear','left_eye','left_ear','right_b_toe','right_s_toe','right_hill','left_b_toe','left_s_toe','left_hill']
        target_connections = [[0,1],[1,2],[1,3],[3,4],[4,5],[0,6],[6,7],[7,8],[1,9],[9,10],[10,11],[0,12],[12,13],[13,14],[2,15],[15,16],[2,17],[17,18],[8,19],[19,20],[8,21],
                        [14,22],[22,23],[14,24]]
        color_bones_target = '-b'
        color_joints_target = 'b'
        
        input_labels = ["hip",'right_hip','right_knee','right_ankle','left_hip','left_knee','left_ankle','chest','neck','chin','head',
                        'right_shoulder','right_elbow','right_wrist','left_shoulder','left_elbow','left_wrist']
        input_connections = [[0,1],[1,2],[2,3],[0,4],[4,5],[5,6],[0,7],[7,8],[8,9],[9,10],[8,11],[11,12],
                        [12,13],[8,14],[14,15],[15,16]]
        color_bones_input = '-r'
        color_joints_input = 'r'

        joints_keep = [0,1,2,3,4,5,6,8,9,11,12,13,14,15,16]
        joints_masked = [15,16,17,18,19,20,21,22,23,24]

        bone_map_input = [[0,1],[1,2],[2,3],[0,4],[4,5],[5,6],[8,9],[8,11],[11,12],[12,13],[8,14],[14,15],[15,16]]
        bone_map_target = []
        
        for bone in target_connections:
            for index in joints_masked:
                if index in bone:
                    bone_map_target.append(bone)
                    break

    elif FLAGS.train_dataset == 'PANOPTIC2' and FLAGS.dataset == 'CMU_all':
            
        target_labels = ['pelvis',"neck","nose",'right_shoulder','right_elbow','right_wrist','right_hip','right_knee','right_ankle','left_shoulder','left_elbow','left_wrist',
                    'left_hip','left_knee','left_ankle','right_eye','right_ear','left_eye','left_ear','right_b_toe','right_s_toe','right_hill','left_b_toe','left_s_toe','left_hill']
        target_connections = [[0,1],[1,2],[1,3],[3,4],[4,5],[0,6],[6,7],[7,8],[1,9],[9,10],[10,11],[0,12],[12,13],[13,14],[2,15],[15,16],[2,17],[17,18],[8,19],[19,20],[8,21],
                        [14,22],[22,23],[14,24]]
        color_bones_target = '-b'
        color_joints_target = 'b'
        
        input_labels = ['pelvis','left_hip','right_hip','spine1','left_knee','right_knee','spine2','left_ankle','right_ankle',
                    'spine3','left_foot','right_foot','neck','left_collar','right_collar','head','left_shoulder','right_shoulder',
                    'left_elbow','right_elbow','left_wrist','right_wrist','left_hand','right_hand']
        input_connections = [[0,1],[0,2],[0,3],[1,4],[2,5],[3,6],[4,7],[5,8],[6,9],[7,10],[8,11],[9,12],[9,13],[9,14],[12,15],[13,16],[14,17],
                        [16,18],[17,19],[18,20],[19,21],[20,22],[21,23]]
        color_bones_input = '-g'
        color_joints_input = 'g'

        joints_keep = [0,4,5,7,8,10,11,12,16,17,18,19,20,21]
        joints_masked = [2,6,12,15,16,17,18,20,21,23,24]

        bone_map_input = [[0,1],[0,2],[1,4],[2,5],[9,13],[9,14],[13,16],[14,17],[0,3],[3,6],[6,9],[9,12],[12,15],[21,23],[20,22]]
        bone_map_target = []
        
        for bone in target_connections:
            for index in joints_masked:
                if index in bone:
                    bone_map_target.append(bone)
                    break

    color_common_bones = '-k'
    color_common_joints = 'y'

    if FLAGS.rotate:
        R = z_rotation(np.radians(90))
        input = np.dot(input,R)
        target = np.dot(target,R)

    #PLOT INPUT
    for i,connections in enumerate(input_connections):
        x1,x2,y1,y2,z1,z2 = connect3d(input[:,0], input[:,1],input[:,2],connections[0],connections[1])
        ax1.plot([x1,x2],[y1,y2],[z1,z2],color_bones_input)#,linewidth=8)
    for i,l in enumerate(input_labels):
        ax1.scatter(input[i,0], input[i,1],input[i,2],color=color_joints_input)#,s=500)
    
    #PLOT INPUT + TARGET

    for i,connections in enumerate(input_connections):
        if connections not in bone_map_input:
            x1,x2,y1,y2,z1,z2 = connect3d(input[:,0], input[:,1],input[:,2],connections[0],connections[1])
            ax.plot([x1,x2],[y1,y2],[z1,z2], color_common_bones)#,linewidth=8)
        else:
            x1,x2,y1,y2,z1,z2 = connect3d(input[:,0], input[:,1],input[:,2],connections[0],connections[1])
            ax.plot([x1,x2],[y1,y2],[z1,z2],color_bones_input)#,linewidth=8)

    for i,l in enumerate(input_labels):
        if i not in joints_keep:
            ax.scatter(input[i,0], input[i,1],input[i,2],color=color_joints_input)#,s=500)
        else:
            ax.scatter(input[i,0], input[i,1],input[i,2],color=color_common_joints,edgecolors='black')#,linewidth=3),s=500)
        

    for i,connections in enumerate(target_connections):
        if connections not in bone_map_target:
            x1,x2,y1,y2,z1,z2 = connect3d(target[:,0], target[:,1],target[:,2],connections[0],connections[1])
            ax.plot([x1,x2],[y1,y2],[z1,z2],color_common_bones)#,linewidth=8)
        else:
            x1,x2,y1,y2,z1,z2 = connect3d(target[:,0], target[:,1],target[:,2],connections[0],connections[1])
            ax.plot([x1,x2],[y1,y2],[z1,z2],color_bones_target)#,linewidth=8)

    for i,l in enumerate(target_labels):
        if i not in joints_masked:
            ax.scatter(target[i,0], target[i,1],target[i,2],color=color_common_joints,edgecolors='black')#,linewidth=3),s=500)
        else:
            ax.scatter(target[i,0], target[i,1],target[i,2],color=color_joints_target)#,s=500)

    #PLOT TARGET
    for i,connections in enumerate(target_connections):
        x1,x2,y1,y2,z1,z2 = connect3d(target[:,0], target[:,1],target[:,2],connections[0],connections[1])
        ax2.plot([x1,x2],[y1,y2],[z1,z2],color_bones_target)#,linewidth=8)
    for i,l in enumerate(target_labels):
        ax2.scatter(target[i,0], target[i,1],target[i,2],color=color_joints_target)#,s=500)


    # labels_all = ["hip",'right_hip','right_knee','right_ankle', 'right_toe','left_hip','left_knee','left_ankle','left_toe','chest','neck','chin','head',
    #                 'right_shoulder','right_elbow','right_wrist', 'right_hand','left_shoulder','left_elbow','left_wrist', 'left_hand']
    # connections_all = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],[9,10],[10,11],[11,12],[10,13],[13,14],
    #                 [14,15],[15,16],[10,17],[17,18],[18,19],[19,20]]
    color_bones = '-r'
    color_joints = 'r'


    # R = x_rotation(np.radians(90))
    # input = np.dot(input,R)

    labels_all = ['pelvis','left_hip','right_hip','spine1','left_knee','right_knee','spine2','left_ankle','right_ankle',
                    'spine3','left_foot','right_foot','neck','left_collar','right_collar','head','left_shoulder','right_shoulder',
                    'left_elbow','right_elbow','left_wrist','right_wrist','left_hand','right_hand']
    connections_all = [[0,1],[0,2],[0,3],[1,4],[2,5],[3,6],[4,7],[5,8],[6,9],[7,10],[8,11],[9,12],[9,13],[9,14],[12,15],[13,16],[14,17],
                    [16,18],[17,19],[18,20],[19,21],[20,22],[21,23]]
    color_bones_input = '-b'
    color_joints_input = 'b'
        
    for i,connections in enumerate(connections_all):
        x1,x2,y1,y2,z1,z2 = connect3d(input_all[:,0], input_all[:,1],input_all[:,2],connections[0],connections[1])
        ax3.plot([x1,x2],[y1,y2],[z1,z2],color_bones)
    for i,l in enumerate(labels_all):
        ax3.scatter(input_all[i,0], input_all[i,1],input_all[i,2],color=color_joints)


    ax3.set_title('Input All',fontsize=10)
    ax3.set_xlim([-0.1, 0.1])
    ax3.set_ylim([-0.1, 0.1])
    ax3.set_zlim([-0.1, 0.1])
    ax3.w_xaxis.pane.fill = False
    ax3.w_yaxis.pane.fill = False
    ax3.w_zaxis.pane.fill = False
    ax3.set_xticks([]) 
    ax3.set_yticks([]) 
    ax3.set_zticks([])
    ax3.set_axis_off()

    ax.set_title('Domain Adaptation',fontsize=10)
    ax.set_xlim([-0.1, 0.1])
    ax.set_ylim([-0.1, 0.1])
    ax.set_zlim([-0.1, 0.1])
    ax.w_xaxis.pane.fill = False
    ax.w_yaxis.pane.fill = False
    ax.w_zaxis.pane.fill = False
    ax.set_xticks([]) 
    ax.set_yticks([]) 
    ax.set_zticks([])
    ax.set_axis_off()
    ax1.set_title('Input',fontsize=10)
    ax1.set_xlim([-0.1, 0.1])
    ax1.set_ylim([-0.1, 0.1])
    ax1.set_zlim([-0.1, 0.1])
    ax1.set_xticks([]) 
    ax1.set_yticks([]) 
    ax1.set_zticks([])
    ax1.w_xaxis.pane.fill = False
    ax1.w_yaxis.pane.fill = False
    ax1.w_zaxis.pane.fill = False
    ax1.set_axis_off()
    ax2.set_title('Target',fontsize=10)
    ax2.set_xlim([-0.1, 0.1])
    ax2.set_ylim([-0.1, 0.1])
    ax2.set_zlim([-0.1, 0.1])
    ax2.w_xaxis.pane.fill = False
    ax2.w_yaxis.pane.fill = False
    ax2.w_zaxis.pane.fill = False
    ax2.set_xticks([]) 
    ax2.set_yticks([]) 
    ax2.set_zticks([])
    ax2.set_axis_off()

    if show:
        plt.show()

    # if FLAGS.rotate:
    #     fig.savefig('/home/giuliamartinelli/Documents/Code/UnsupervisedHMR/DomainAdaptationModule/Test/%s_%s_%s_rot' %(FLAGS.dataset,FLAGS.train_dataset,idx),transparent=True)
    # else:
    #     fig.savefig('/home/giuliamartinelli/Documents/Code/UnsupervisedHMR/DomainAdaptationModule/Test/%s_%s_%s' %(FLAGS.dataset,FLAGS.train_dataset,idx),transparent=True)

def MPJPE(skeleton1, skeleton2):
    difference=[]

    if skeleton1.shape == skeleton2.shape:
        for i,joint in enumerate(skeleton1):
            difference.append(np.linalg.norm(skeleton1[i] - skeleton2[i]))

    res=0
    for x in difference:
        res+=x

    return res/len(skeleton1)

def normalize_head3d(poses_3d, root_joint=0,type='smpl'):
    # center at root joint
    if type == 'smpl':
        p2d = poses_3d.reshape(-1, 3, 24)
        p2d -= p2d[:, :, [root_joint]]
        scale = np.linalg.norm(p2d[:, :, 0] - p2d[:, :, 15], axis=1, keepdims=True)
        p2ds = p2d / scale.mean()
        p2ds = p2ds * (1 / 10)
        p2ds = p2ds.reshape(-1,72)
    elif type== 'h36m':
        p2d = poses_3d.reshape(-1, 3, 17)
        p2d -= p2d[:, :, [root_joint]]
        scale = np.linalg.norm(p2d[:, :, 0] - p2d[:, :, 10], axis=1, keepdims=True)
        p2ds = p2d / scale.mean()
        p2ds = p2ds * (1 / 10)
        p2ds = p2ds.reshape(-1,51)
    elif type== 'h36m_all':
        p2d = poses_3d.reshape(-1, 3, 21)
        p2d -= p2d[:, :, [root_joint]]
        scale = np.linalg.norm(p2d[:, :, 0] - p2d[:, :, 12], axis=1, keepdims=True)
        p2ds = p2d / scale.mean()
        p2ds = p2ds * (1 / 10)
        p2ds = p2ds.reshape(-1,63)
    else:
        p2d = poses_3d.reshape(-1, 3, 25)
        p2d -= p2d[:, :, [root_joint]]
        scale = np.linalg.norm(p2d[:, :, 0] - p2d[:, :, 2], axis=1, keepdims=True)
        p2ds = p2d / scale.mean()
        p2ds = p2ds * (1 / 10)
        p2ds = p2ds.reshape(-1,75)

    return p2ds

def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_uniform_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.kaiming_uniform_(m.weight.data)
    elif classname.find('BatchNorm') != -1:
        nn.init.ones_(m.weight.data)
        nn.init.zeros_(m.bias.data)

def weights_init_trunc(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.trunc_normal_(m.weight.data, std=.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.trunc_normal_(m.weight.data, std=.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.ones_(m.weight.data)
        nn.init.zeros_(m.bias.data)

def flip_joints(joints, flip_pairs, dimension=3,axis = 'x',type = 'smpl'): #"b (d j) -> b j d"
    joints = rearrange(joints, "(d j) -> j d", d = dimension)
    joints_flipped = torch.empty(size=joints.shape)
    for i in range(joints_flipped.shape[0]):
        for j in range(3):
            if axis == 'x':
                if j == 0:
                    joints_flipped[i, j] = -joints[i, j]
                else:
                    joints_flipped[i, j] = joints[i, j]
            if axis == 'z' and type == 'smpl':
                if j == 1:
                    joints_flipped[i, j] = -joints[i, j]
                else:
                    joints_flipped[i, j] = joints[i, j]
            if axis == 'z' and type == 'h36m':
                if j == 2:
                    joints_flipped[i, j] = -joints[i, j]
                else:
                    joints_flipped[i, j] = joints[i, j]
    
    for left, right in flip_pairs:
        for j in range(3):
            if axis == 'x':
                if j == 0:
                    joints_flipped[left][j] = -joints[right][j]
                    joints_flipped[right][j] = -joints[left][j]
                else:
                    joints_flipped[left][j] = joints[right][j]
                    joints_flipped[right][j] = joints[left][j]
            if axis == 'z' and type == 'smpl':
                if j == 1:
                    joints_flipped[left][j] = -joints[right][j]
                    joints_flipped[right][j] = -joints[left][j]
                else:
                    joints_flipped[left][j] = joints[right][j]
                    joints_flipped[right][j] = joints[left][j]
            if axis == 'z' and type == 'h36m':
                if j == 2:
                    joints_flipped[left][j] = -joints[right][j]
                    joints_flipped[right][j] = -joints[left][j]
                else:
                    joints_flipped[left][j] = joints[right][j]
                    joints_flipped[right][j] = joints[left][j]

    joints_flipped = rearrange(joints_flipped, "j d -> (d j)")

    return joints_flipped

def get_bone_lengths_all(poses, n_joints=17, bone_map=None,type='torch'):
    
    if bone_map == None:
        bone_map = [[0,1],[0,2],[0,3],[1,4],[2,5],[3,6],[4,7],[5,8],[6,9],[7,10],[8,11],[9,12],[9,13],[9,14],[12,15],[13,16],[14,17],
                            [16,18],[17,19],[18,20],[19,21],[20,22],[21,23]]

    poses = poses.reshape((-1, 3, n_joints))

    ext_bones = poses[:, :, bone_map]

    bones = ext_bones[:, :, :, 0] - ext_bones[:, :, :, 1]

    if type == 'torch':
        bone_lengths = torch.norm(bones, p=2, dim=1)
    else:
        bone_lengths = np.linalg.norm(bones, ord=2, axis=1)

    return bone_lengths

def load_model(pose2rot,modelFolder,numBetas):
    
    # model_male = smplx.create(modelFolder, model_type='smpl',
    #                             gender='male',
    #                             ext='npz')
    # model_female = smplx.create(modelFolder, model_type='smpl',
    #                             gender='female',
    #                             ext='npz')
    model_neutral = smplx.create(modelFolder, model_type='smpl',
                                    gender='neutral',
                                    ext='pkl')
    return model_neutral

