import numpy as np
import cv2
import os, sys, time
from tqdm import tqdm
import glob
import argparse
import torch
import torch.nn.functional as F
import torch.nn as nn
import cv2

import torchvision.models as models
from  svcca import cca_core
#from models import *
import pickle


def assess_a_clip(model, frame_clip):
    features = []
    attrs = []
    for i in range (frame_clip.shape[0]):
        feature, attr = model(frame_clip[i])
        features.append(feature.detach().cpu().numpy())
        attrs.append(attr.detach().cpu().numpy())
    
    features = np.concatenate(features,0)
    attrs = np.concatenate(attrs)

    return features, attrs


def compute_cca( feat1, feat2):
    #num_datapoints, h, w, c = feat1.shape
    #feat1.reshape(num_datapoints*h*w, c)
    #num_datapoints, h, w, c = feat2.shape
   # feat2.reshape(num_datapoints*h*w, c)
    #print(feat1.shape, feat2.shape)
    
    f_result = cca_core.get_cca_similarity(feat1.transpose(1,0).detach().cpu().numpy(), 
                                           feat2.transpose(1,0).detach().cpu().numpy(), epsilon=1e-10, verbose=False)

    mean_cca = np.mean(f_result["cca_coef1"])
    return mean_cca, f_result


def gen_A_from_clips( model, frame_clips, distype='cca'):
    node_num = len(frame_clips)
    adj = np.empty((node_num, node_num), dtype=float)
    for i in range(node_num):
        j = i+1
        while(j<node_num):
            feature_i, attrs_i = assess_a_clip(model, frame_clips[i])
            feature_j, attrs_j = assess_a_clip(model, frame_clips[j])
            if distype == 'cos':
                d = torch.cosine_similarity(attrs_i, attrs_j) # cosine distance #direct compare the attribution level difference
            #eucDis = F.pairwise_distance(feature_i, feature_j) # euclidean distance
            elif distype == 'cca':
                d, _ = compute_cca(feature_i, feature_j) # svcca distance # compare the feature level difference
            else:
                print("Unsurpported Distance type:", distype)
                sys.exit()

            adj[i][j] = d
            adj[j][i] = d

            j = j+1
    
    return adj

def gen_A_from_frames(model, frame_clips, device, num_frames=16, distype='cca'):
    #print(frame_clips.shape)
    frame_clips = frame_clips.permute((0, 2, 3, 4, 1)) # change from 4x3x16x112x112 => 4x16x3x112x112
    # num_frames: number of frames per clip
    node_num = num_frames
    adj = np.zeros((frame_clips.shape[0], node_num, node_num), np.dtype('float32'))
    A_feat = np.zeros((frame_clips.shape[0], node_num, 1000), np.dtype('float32'))
    for i in range(frame_clips.shape[0]):
        features = []
        for k in range(node_num):
            j = k+1
            while(j<node_num):
                fk = frame_clips[i,k,:].detach().cpu().numpy()
                fj = frame_clips[i,j,:].detach().cpu().numpy()
                fk = torch.from_numpy(cv2.resize(fk, (224, 224))).to(device)
                fj = torch.from_numpy(cv2.resize(fj, (224, 224))).to(device)
                fk, fj = fk.permute(2,0,1), fj.permute(2,0,1)
                #print(fk.shape, fj.shape)
                feature_k = model(fk.unsqueeze(0)).transpose(0,1)
                feature_j = model(fj.unsqueeze(0)).transpose(0,1)
                if distype == 'cos':
                    d = torch.cosine_similarity(feature_k, feature_j) # cosine distance #direct compare the attribution level difference
                    d = d.mean()
                #eucDis = F.pairwise_distance(feature_i, feature_j) # euclidean distance
                elif distype == 'cca':
                    d, _ = compute_cca(feature_k, feature_j) # svcca distance # compare the feature level difference
                    #print(d)
                else:
                    print("Unsurpported Distance type:", distype)
                    sys.exit()

                adj[i][k][j] = d
                adj[i][j][k] = d

                j = j+1
            A_feat[i,k,:] = (feature_k.squeeze(1)).detach().cpu().numpy()              
    A_feat =  torch.from_numpy(A_feat).to(device)    
    #print(adj)   
    return A_feat, adj

def gen_A_resnet50(clip_path, device, distype='cca'):
    model = models.resnet50(pretrained = True)
    model.eval()
    model.to(device)
    with torch.no_grad():
        adj = gen_A_from_frames(model, clip_path,device, distype = distype)
    return adj 

def gen_A(num_node, adj_file):
    adj_f = pickle.load(open(adj_file, "rb"))
    mean_cca = adj_f["mean_cca"]
    adj = adj_f["adj"]
    #for i in range ()
    return mean_cca, adj


def gen_adj(A):
    adj = torch.ones_like(A)
    for i in range(A.shape[0]):
        D = torch.pow(A[i,:].sum(1).float(), -0.5)
        D = torch.diag(D)
        adj[i, :] = torch.matmul(torch.matmul(A[i,:], D).t(),D)
    return adj

            





    






    