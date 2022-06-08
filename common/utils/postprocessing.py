import numpy as np
import cv2
import random
from config import cfg
import math
from utils.human_models import smpl, mano, flame
from utils.transforms import cam2pixel, transform_joint_to_other_db
from plyfile import PlyData, PlyElement
import torch


def transform_human_model_parameter_to_2d_image(human_model_param, cam_param, do_flip, img_shape, img2bb_trans, rot, human_model_type):

    if human_model_type == 'smpl':
        human_model = smpl
        pose, shape, trans = human_model_param['pose'], human_model_param['shape'], human_model_param['trans']
        if 'gender' in human_model_param:
            gender = human_model_param['gender']
        else:
            gender = 'neutral'
        pose = torch.FloatTensor(pose).view(-1,3); shape = torch.FloatTensor(shape).view(1,-1); # smpl parameters (pose: 72 dimension, shape: 10 dimension)
        trans = torch.FloatTensor(trans).view(1,-1) # translation vector
        
        # apply camera extrinsic (rotation)
        # merge root pose and camera rotation 
        if 'R' in cam_param:
            R = np.array(cam_param['R'], dtype=np.float32).reshape(3,3)
            root_pose = pose[smpl.orig_root_joint_idx,:].numpy()
            root_pose, _ = cv2.Rodrigues(root_pose)
            root_pose, _ = cv2.Rodrigues(np.dot(R,root_pose))
            pose[smpl.orig_root_joint_idx] = torch.from_numpy(root_pose).view(3)

        # get mesh and joint coordinates
        root_pose = pose[smpl.orig_root_joint_idx].view(1,3)
        body_pose = torch.cat((pose[:smpl.orig_root_joint_idx,:], pose[smpl.orig_root_joint_idx+1:,:])).view(1,-1)
        with torch.no_grad():
            output = smpl.layer[gender](betas=shape, body_pose=body_pose, global_orient=root_pose, transl=trans)
        mesh_coord = output.vertices[0].numpy()
        joint_coord = np.dot(smpl.joint_regressor, mesh_coord)
 
        # apply camera exrinsic (translation)
        # compenstate rotation (translation from origin to root joint was not cancled)
        if 'R' in cam_param and 't' in cam_param:
            R, t = np.array(cam_param['R'], dtype=np.float32).reshape(3,3), np.array(cam_param['t'], dtype=np.float32).reshape(1,3)
            root_coord = joint_coord[smpl.root_joint_idx,None,:]
            joint_coord = joint_coord - root_coord + np.dot(R, root_coord.transpose(1,0)).transpose(1,0) + t
            mesh_coord = mesh_coord - root_coord + np.dot(R, root_coord.transpose(1,0)).transpose(1,0) + t

    elif human_model_type == 'mano':
        human_model = mano
        pose, shape, trans = human_model_param['pose'], human_model_param['shape'], human_model_param['trans']
        hand_type = human_model_param['hand_type']
        trans = human_model_param['trans']
        pose = torch.FloatTensor(pose).view(-1,3); shape = torch.FloatTensor(shape).view(1,-1); # mano parameters (pose: 48 dimension, shape: 10 dimension)
        trans = torch.FloatTensor(trans).view(1,-1) # translation vector

        # apply camera extrinsic (rotation)
        # merge root pose and camera rotation 
        if 'R' in cam_param:
            R = np.array(cam_param['R'], dtype=np.float32).reshape(3,3)
            root_pose = pose[mano.orig_root_joint_idx,:].numpy()
            root_pose, _ = cv2.Rodrigues(root_pose)
            root_pose, _ = cv2.Rodrigues(np.dot(R,root_pose))
            pose[mano.orig_root_joint_idx] = torch.from_numpy(root_pose).view(3)
      
        # get root joint coordinate
        root_pose = pose[mano.orig_root_joint_idx].view(1,3)
        hand_pose = torch.cat((pose[:mano.orig_root_joint_idx,:], pose[mano.orig_root_joint_idx+1:,:])).view(1,-1)
        with torch.no_grad():
            output = mano.layer[hand_type](betas=shape, hand_pose=hand_pose, global_orient=root_pose, transl=trans)
        mesh_coord = output.vertices[0].numpy()
        joint_coord = np.dot(mano.joint_regressor, mesh_coord)
      
        # apply camera exrinsic (translation)
        # compenstate rotation (translation from origin to root joint was not cancled)
        if 'R' in cam_param and 't' in cam_param:
            R, t = np.array(cam_param['R'], dtype=np.float32).reshape(3,3), np.array(cam_param['t'], dtype=np.float32).reshape(1,3)
            root_coord = joint_coord[mano.root_joint_idx,None,:]
            joint_coord = joint_coord - root_coord + np.dot(R, root_coord.transpose(1,0)).transpose(1,0) + t
            mesh_coord = mesh_coord - root_coord + np.dot(R, root_coord.transpose(1,0)).transpose(1,0) + t

    elif human_model_type == 'flame':
        human_model = flame
        root_pose, jaw_pose, shape, expr = human_model_param['root_pose'], human_model_param['jaw_pose'], human_model_param['shape'], human_model_param['expr']
        if 'trans' in human_model_param:
            trans = human_model_param['trans']
        else:
            trans = [0,0,0]
        root_pose = torch.FloatTensor(root_pose).view(1,3); jaw_pose = torch.FloatTensor(jaw_pose).view(1,3);
        shape = torch.FloatTensor(shape).view(1,-1); expr = torch.FloatTensor(expr).view(1,-1);
        zero_pose = torch.zeros((1,3)).float() # neck and eye poses
        trans = torch.FloatTensor(trans).view(1,-1) # translation vector
 
        # apply camera extrinsic (rotation)
        # merge root pose and camera rotation 
        if 'R' in cam_param:
            R = np.array(cam_param['R'], dtype=np.float32).reshape(3,3)
            root_pose = root_pose.numpy()
            root_pose, _ = cv2.Rodrigues(root_pose)
            root_pose, _ = cv2.Rodrigues(np.dot(R,root_pose))
            root_pose = torch.from_numpy(root_pose).view(1,3)
      
        # get root joint coordinate
        with torch.no_grad():
            output = flame.layer(global_orient=root_pose, jaw_pose=jaw_pose, neck_pose=zero_pose, leye_pose=zero_pose, reye_pose=zero_pose, betas=shape, expression=expr, transl=trans)
        mesh_coord = output.vertices[0].numpy()
        joint_coord = output.joints[0].numpy()
      
        # apply camera exrinsic (translation)
        # compenstate rotation (translation from origin to root joint was not cancled)
        if 'R' in cam_param and 't' in cam_param:
            R, t = np.array(cam_param['R'], dtype=np.float32).reshape(3,3), np.array(cam_param['t'], dtype=np.float32).reshape(1,3)
            root_coord = joint_coord[flame.root_joint_idx,None,:]
            joint_coord = joint_coord - root_coord + np.dot(R, root_coord.transpose(1,0)).transpose(1,0) + t
            mesh_coord = mesh_coord - root_coord + np.dot(R, root_coord.transpose(1,0)).transpose(1,0) + t
    
    joint_cam_orig = joint_coord.copy() # back-up the original one
    mesh_cam_orig = mesh_coord.copy() # back-up the original one

    ## so far, joint coordinates are in camera-centered 3D coordinates (data augmentations are not applied yet)
    ## now, project the 3D coordinates to image space and apply data augmentations

    # image projection
    joint_cam = joint_coord # camera-centered 3D coordinates
    joint_img = cam2pixel(joint_cam, cam_param['focal'], cam_param['princpt'])
    joint_cam = joint_cam - joint_cam[human_model.root_joint_idx,None,:] # root-relative
    joint_img[:,2] = joint_cam[:,2].copy()
    if do_flip:
        joint_cam[:,0] = -joint_cam[:,0]
        joint_img[:,0] = img_shape[1] - 1 - joint_img[:,0]
        for pair in human_model.flip_pairs:
            joint_cam[pair[0], :], joint_cam[pair[1], :] = joint_cam[pair[1], :].copy(), joint_cam[pair[0], :].copy()
            joint_img[pair[0], :], joint_img[pair[1], :] = joint_img[pair[1], :].copy(), joint_img[pair[0], :].copy()

    # x,y affine transform, root-relative depth
    joint_img_xy1 = np.concatenate((joint_img[:,:2], np.ones_like(joint_img[:,0:1])),1)
    joint_img[:,:2] = np.dot(img2bb_trans, joint_img_xy1.transpose(1,0)).transpose(1,0)[:,:2]
    joint_img[:,0] = joint_img[:,0] / cfg.input_img_shape[1] * cfg.output_hm_shape[2]
    joint_img[:,1] = joint_img[:,1] / cfg.input_img_shape[0] * cfg.output_hm_shape[1]
    joint_img[:,2] = (joint_img[:,2] / (cfg.bbox_3d_size / 2) + 1)/2. * cfg.output_hm_shape[0]
    
    # check truncation
    joint_trunc = ((joint_img[:,0] >= 0) * (joint_img[:,0] < cfg.output_hm_shape[2]) * \
                (joint_img[:,1] >= 0) * (joint_img[:,1] < cfg.output_hm_shape[1]) * \
                (joint_img[:,2] >= 0) * (joint_img[:,2] < cfg.output_hm_shape[0])).reshape(-1,1).astype(np.float32)
    
    # 3D data rotation augmentation
    rot_aug_mat = np.array([[np.cos(np.deg2rad(-rot)), -np.sin(np.deg2rad(-rot)), 0], 
    [np.sin(np.deg2rad(-rot)), np.cos(np.deg2rad(-rot)), 0],
    [0, 0, 1]], dtype=np.float32)
    # coordinate
    joint_cam = np.dot(rot_aug_mat, joint_cam.transpose(1,0)).transpose(1,0)
    # parameters
    if human_model_type == 'flame':
        # flip pose parameter (axis-angle)
        if do_flip:
            root_pose[:,1:3] *= -1 # multiply -1 to y and z axis of axis-angle
            jaw_pose[:,1:3] *= -1
        # rotate root pose
        root_pose = root_pose.numpy()
        root_pose, _ = cv2.Rodrigues(root_pose)
        root_pose, _ = cv2.Rodrigues(np.dot(rot_aug_mat,root_pose))
        root_pose = root_pose.reshape(-1)
    else:
        # flip pose parameter (axis-angle)
        if do_flip:
            for pair in human_model.orig_flip_pairs:
                pose[pair[0], :], pose[pair[1], :] = pose[pair[1], :].clone(), pose[pair[0], :].clone()
            pose[:,1:3] *= -1 # multiply -1 to y and z axis of axis-angle
        # rotate root pose
        pose = pose.numpy()
        root_pose = pose[human_model.orig_root_joint_idx,:]
        root_pose, _ = cv2.Rodrigues(root_pose)
        root_pose, _ = cv2.Rodrigues(np.dot(rot_aug_mat,root_pose))
        pose[human_model.orig_root_joint_idx] = root_pose.reshape(3)
    
    # return results
    if human_model_type == 'flame':
        jaw_pose = jaw_pose.numpy().reshape(-1)
        # change to mean shape if beta is too far from it
        shape[(shape.abs() > 3).any(dim=1)] = 0.
        shape = shape.numpy().reshape(-1)
        expr = expr.numpy().reshape(-1)
        return joint_img, joint_cam, joint_trunc, root_pose, jaw_pose, shape, expr, joint_cam_orig, mesh_cam_orig
    else:
        pose = pose.reshape(-1)
        # change to mean shape if beta is too far from it
        shape[(shape.abs() > 3).any(dim=1)] = 0.
        shape = shape.numpy().reshape(-1)
        return joint_img, joint_cam, joint_trunc, pose, shape, mesh_cam_orig

def transform_human_model_parameter_to_2d_image11(human_model_param, cam_param, human_model_type):

    if human_model_type == 'smpl':
        human_model = smpl
        pose, shape, trans = human_model_param['pose'], human_model_param['shape'], human_model_param['trans']
        if 'gender' in human_model_param:
            gender = human_model_param['gender']
        else:
            gender = 'neutral'
        pose = torch.FloatTensor(pose).view(-1,3) 
        shape = torch.FloatTensor(shape).view(1,-1) # smpl parameters (pose: 72 dimension, shape: 10 dimension)
        trans = torch.FloatTensor(trans).view(1,-1) # translation vector
        
        # apply camera extrinsic (rotation)
        # merge root pose and camera rotation 
        if 'R' in cam_param:
            R = np.array(cam_param['R'], dtype=np.float32).reshape(3,3)
            root_pose = pose[smpl.orig_root_joint_idx,:].numpy()
            root_pose, _ = cv2.Rodrigues(root_pose)
            root_pose, _ = cv2.Rodrigues(np.dot(R,root_pose))
            pose[smpl.orig_root_joint_idx] = torch.from_numpy(root_pose).view(3)

        # get mesh and joint coordinates
        root_pose = pose[smpl.orig_root_joint_idx].view(1,3)
        body_pose = torch.cat((pose[:smpl.orig_root_joint_idx,:], pose[smpl.orig_root_joint_idx+1:,:])).view(1,-1)
        with torch.no_grad():
            output = smpl.layer[gender](betas=shape, body_pose=body_pose, global_orient=root_pose, transl=trans)
        mesh_coord = output.vertices[0].numpy()
        joint_coord = np.dot(smpl.joint_regressor, mesh_coord)
 
        # apply camera exrinsic (translation)
        # compenstate rotation (translation from origin to root joint was not cancled)
        if 'R' in cam_param and 't' in cam_param:
            R, t = np.array(cam_param['R'], dtype=np.float32).reshape(3,3), np.array(cam_param['t'], dtype=np.float32).reshape(1,3)
            root_coord = joint_coord[smpl.root_joint_idx,None,:]
            joint_coord = joint_coord - root_coord + np.dot(R, root_coord.transpose(1,0)).transpose(1,0) + t
            mesh_coord = mesh_coord - root_coord + np.dot(R, root_coord.transpose(1,0)).transpose(1,0) + t

    elif human_model_type == 'mano':
        human_model = mano
        pose, shape, trans = human_model_param['pose'], human_model_param['shape'], human_model_param['trans']
        hand_type = human_model_param['hand_type']
        trans = human_model_param['trans']
        pose = torch.FloatTensor(pose).view(-1,3); shape = torch.FloatTensor(shape).view(1,-1); # mano parameters (pose: 48 dimension, shape: 10 dimension)
        trans = torch.FloatTensor(trans).view(1,-1) # translation vector

        # apply camera extrinsic (rotation)
        # merge root pose and camera rotation 
        if 'R' in cam_param:
            R = np.array(cam_param['R'], dtype=np.float32).reshape(3,3)
            root_pose = pose[mano.orig_root_joint_idx,:].numpy()
            root_pose, _ = cv2.Rodrigues(root_pose)
            root_pose, _ = cv2.Rodrigues(np.dot(R,root_pose))
            pose[mano.orig_root_joint_idx] = torch.from_numpy(root_pose).view(3)
      
        # get root joint coordinate
        root_pose = pose[mano.orig_root_joint_idx].view(1,3)
        hand_pose = torch.cat((pose[:mano.orig_root_joint_idx,:], pose[mano.orig_root_joint_idx+1:,:])).view(1,-1)
        with torch.no_grad():
            output = mano.layer[hand_type](betas=shape, hand_pose=hand_pose, global_orient=root_pose, transl=trans)
        mesh_coord = output.vertices[0].numpy()
        joint_coord = np.dot(mano.joint_regressor, mesh_coord)
      
        # apply camera exrinsic (translation)
        # compenstate rotation (translation from origin to root joint was not cancled)
        if 'R' in cam_param and 't' in cam_param:
            R, t = np.array(cam_param['R'], dtype=np.float32).reshape(3,3), np.array(cam_param['t'], dtype=np.float32).reshape(1,3)
            root_coord = joint_coord[mano.root_joint_idx,None,:]
            joint_coord = joint_coord - root_coord + np.dot(R, root_coord.transpose(1,0)).transpose(1,0) + t
            mesh_coord = mesh_coord - root_coord + np.dot(R, root_coord.transpose(1,0)).transpose(1,0) + t

    elif human_model_type == 'flame':
        human_model = flame
        root_pose, jaw_pose, shape, expr = human_model_param['root_pose'], human_model_param['jaw_pose'], human_model_param['shape'], human_model_param['expr']
        if 'trans' in human_model_param:
            trans = human_model_param['trans']
        else:
            trans = [0,0,0]
        root_pose = torch.FloatTensor(root_pose).view(1,3); jaw_pose = torch.FloatTensor(jaw_pose).view(1,3);
        shape = torch.FloatTensor(shape).view(1,-1); expr = torch.FloatTensor(expr).view(1,-1);
        zero_pose = torch.zeros((1,3)).float() # neck and eye poses
        trans = torch.FloatTensor(trans).view(1,-1) # translation vector
 
        # apply camera extrinsic (rotation)
        # merge root pose and camera rotation 
        if 'R' in cam_param:
            R = np.array(cam_param['R'], dtype=np.float32).reshape(3,3)
            root_pose = root_pose.numpy()
            root_pose, _ = cv2.Rodrigues(root_pose)
            root_pose, _ = cv2.Rodrigues(np.dot(R,root_pose))
            root_pose = torch.from_numpy(root_pose).view(1,3)
      
        # get root joint coordinate
        with torch.no_grad():
            output = flame.layer(global_orient=root_pose, jaw_pose=jaw_pose, neck_pose=zero_pose, leye_pose=zero_pose, reye_pose=zero_pose, betas=shape, expression=expr, transl=trans)
        mesh_coord = output.vertices[0].numpy()
        joint_coord = output.joints[0].numpy()
      
        # apply camera exrinsic (translation)
        # compenstate rotation (translation from origin to root joint was not cancled)
        if 'R' in cam_param and 't' in cam_param:
            R, t = np.array(cam_param['R'], dtype=np.float32).reshape(3,3), np.array(cam_param['t'], dtype=np.float32).reshape(1,3)
            root_coord = joint_coord[flame.root_joint_idx,None,:]
            joint_coord = joint_coord - root_coord + np.dot(R, root_coord.transpose(1,0)).transpose(1,0) + t
            mesh_coord = mesh_coord - root_coord + np.dot(R, root_coord.transpose(1,0)).transpose(1,0) + t
    
    joint_cam_orig = joint_coord.copy() # back-up the original one
    mesh_cam_orig = mesh_coord.copy() # back-up the original one

    ## so far, joint coordinates are in camera-centered 3D coordinates (data augmentations are not applied yet)
    ## now, project the 3D coordinates to image space and apply data augmentations

    # image projection
    joint_cam = joint_coord # camera-centered 3D coordinates
    joint_img = cam2pixel(joint_cam, cam_param['focal'], cam_param['princpt'])
    joint_cam = joint_cam - joint_cam[human_model.root_joint_idx,None,:] # root-relative
    joint_img[:,2] = joint_cam[:,2].copy()
    
    joint_img[:,0] = joint_img[:,0] / cfg.input_img_shape[1] * cfg.output_hm_shape[2]
    joint_img[:,1] = joint_img[:,1] / cfg.input_img_shape[0] * cfg.output_hm_shape[1]
    joint_img[:,2] = (joint_img[:,2] / (cfg.bbox_3d_size / 2) + 1)/2. * cfg.output_hm_shape[0]
    
    # check truncation
    joint_trunc = ((joint_img[:,0] >= 0) * (joint_img[:,0] < cfg.output_hm_shape[2]) * \
                (joint_img[:,1] >= 0) * (joint_img[:,1] < cfg.output_hm_shape[1]) * \
                (joint_img[:,2] >= 0) * (joint_img[:,2] < cfg.output_hm_shape[0])).reshape(-1,1).astype(np.float32)
    
    if human_model_type == 'flame':
        root_pose = root_pose.numpy()
        #root_pose, _ = cv2.Rodrigues(root_pose)
        #root_pose, _ = cv2.Rodrigues(np.dot(rot_aug_mat,root_pose))
        root_pose = root_pose.reshape(-1)
    else:
        pose = pose.numpy()
        root_pose = pose[human_model.orig_root_joint_idx,:]
        #root_pose, _ = cv2.Rodrigues(root_pose)
        #root_pose, _ = cv2.Rodrigues(np.dot(rot_aug_mat,root_pose))
        pose[human_model.orig_root_joint_idx] = root_pose.reshape(3)
    
    # return results
    if human_model_type == 'flame':
        jaw_pose = jaw_pose.numpy().reshape(-1)
        # change to mean shape if beta is too far from it
        shape[(shape.abs() > 3).any(dim=1)] = 0.
        shape = shape.numpy().reshape(-1)
        expr = expr.numpy().reshape(-1)
        return joint_img, joint_cam, joint_trunc, root_pose, jaw_pose, shape, expr, joint_cam_orig, mesh_cam_orig
    else:
        pose = pose.reshape(-1)
        # change to mean shape if beta is too far from it
        shape[(shape.abs() > 3).any(dim=1)] = 0.
        shape = shape.numpy().reshape(-1)
        return joint_img, joint_cam, joint_trunc, pose, shape, mesh_cam_orig
