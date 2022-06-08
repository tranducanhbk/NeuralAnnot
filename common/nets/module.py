# import os.path as osp
import torch
import torch.nn as nn
from torch.nn import functional as F
from nets.layer import make_conv_layers, make_linear_layers
from utils.human_models import smpl, mano, flame
from utils.transforms import sample_joint_features, soft_argmax_3d
from config import cfg
import numpy as np

class PositionNet(nn.Module):
    def __init__(self):
        super(PositionNet, self).__init__()
        if cfg.parts == 'body':
            self.joint_num = smpl.pos_joint_num
        elif cfg.parts == 'hand':
            self.joint_num = mano.joint_num
        self.conv = make_conv_layers([2048,self.joint_num*cfg.output_hm_shape[0]], kernel=1, stride=1, padding=0, bnrelu_final=False)
        

    def forward(self, img_feat):
        joint_hm = self.conv(img_feat).view(-1,self.joint_num,cfg.output_hm_shape[0],cfg.output_hm_shape[1],cfg.output_hm_shape[2])
        joint_coord = soft_argmax_3d(joint_hm)
        return joint_coord

class RotationNet(nn.Module):
    def __init__(self):
        super(RotationNet, self).__init__()
        if cfg.parts == 'body':
            self.joint_num = smpl.pos_joint_num
        elif cfg.parts == 'hand':
            self.joint_num = mano.joint_num
       
        # output layers
        if cfg.parts == 'body':
            #self.conv = make_conv_layers([2048,512], kernel=1, stride=1, padding=0)
            #self.root_pose_out = make_linear_layers([self.joint_num*(512+3), 6], relu_final=False)
            #self.pose_out = make_linear_layers([self.joint_num*(512+3), (smpl.orig_joint_num-3)*6], relu_final=False) # without root and two hands
            # predict 32 dismention for vposer
            self.vposer_out = make_linear_layers([2048, 63], relu_final=False)
            # predict global rotaions of smpl 
            self.root_pose_out = make_linear_layers([2048, 3], relu_final=False)
            # predict shape parameters of smpl
            self.shape_out = make_linear_layers([2048,smpl.shape_param_dim], relu_final=False)
            # predict cam parameters of smpl
            self.cam_out = make_linear_layers([2048,3], relu_final=False)
            #self.rotaion_out = make_linear_layers([2048,48], relu_final=False)


        elif cfg.parts == 'hand':
            self.conv = make_conv_layers([2048,512], kernel=1, stride=1, padding=0)
            self.root_pose_out = make_linear_layers([self.joint_num*(512+3), 6], relu_final=False)
            self.pose_out = make_linear_layers([self.joint_num*(512+3), (mano.orig_joint_num-1)*6], relu_final=False) # without root joint
            self.shape_out = make_linear_layers([2048,mano.shape_param_dim], relu_final=False)
            self.cam_out = make_linear_layers([2048,3], relu_final=False)

    def forward(self, img_feat):
        feat = img_feat.mean((2,3))
        #outs = self.rotaion_out(img_feat.mean((2,3)))
        # root pose
        root_pose =  self.root_pose_out(feat)
        # vposer para 
        vposer_para = self.vposer_out(feat)

         # shape parameter
        shape_param =  self.shape_out(feat)

        # camera parameter
        cam_param = self.cam_out(feat)
        
       
        
        return root_pose, shape_param, cam_param, vposer_para

class FaceRegressor(nn.Module):
    def __init__(self):
        super(FaceRegressor, self).__init__()
        self.pose_out = make_linear_layers([2048,12], relu_final=False) # pose parameter
        self.shape_out = make_linear_layers([2048, flame.shape_param_dim], relu_final=False) # shape parameter
        self.expr_out = make_linear_layers([2048, flame.expr_code_dim], relu_final=False) # expression parameter
        self.cam_out = make_linear_layers([2048,3], relu_final=False) # camera parameter

    def forward(self, img_feat):
        feat = img_feat.mean((2,3))
        
        # pose parameter
        pose_param = self.pose_out(feat)
        root_pose = pose_param[:,:6]
        jaw_pose = pose_param[:,6:]

        # shape parameter
        shape_param = self.shape_out(feat)

        # expression parameter
        expr_param = self.expr_out(feat)

        # camera parameter
        cam_param = self.cam_out(feat)

        return root_pose, jaw_pose, shape_param, expr_param, cam_param