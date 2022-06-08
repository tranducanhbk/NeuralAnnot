import sys
import os
import os.path as osp
import argparse
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from torch.nn.parallel.data_parallel import DataParallel
import torch.backends.cudnn as cudnn
from pycocotools.coco import COCO

sys.path.insert(0, osp.join('..', 'main'))
sys.path.insert(0, osp.join('..', 'data'))
sys.path.insert(0, osp.join('..', 'common'))
from config import cfg
from model import get_model
from utils.preprocessing import process_bbox, generate_patch_image, load_img, augmentation, process_db_coord
from utils.human_models import smpl, smpl_x, mano, flame
from utils.vis import render_mesh, save_obj
import json
import time
from nets.loss import CoordLoss
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    args = parser.parse_args()

    # test gpus
    if not args.gpu_ids:
        assert 0, print("Please set proper gpu ids")

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))
    
    return args
def draw_joints(img_copy, body_joint):
    colors = [(122,122,0),(0,255,0),(122,122,0),(122,122,0),(0,255,0),(122,122,0),(122,122,0),(122,122,0),(0,255,0),(122,122,0),(122,122,0),(122,122,0),(0,0,255),(0,0,255),(122,122,0),(122,122,0),(122,122,0),(122,122,0),(122,122,0),(122,122,0),(122,122,0),(122,122,0),(122,122,0),(122,122,0),(122,122,0),(122,122,0),(122,122,0),(122,122,0),(122,122,0),(122,122,0),(122,122,0),(122,122,0),(122,122,0),(122,122,0),(122,122,0),(122,122,0),(122,122,0)]
    print(colors[0])
    for i in range(33):
        img_copy = cv2.circle(img_copy, (int(body_joint[i][0]),int(body_joint[i][1])), 2, colors[i], 2)
    #cv2.imwrite("AAA.jpg",img_copy)
    return img_copy
    
def add_joint(joint_coord, feet_joint_coord, ljoint_coord, rjoint_coord,joint_set):
    # pelvis
    lhip_idx = joint_set['body']['joints_name'].index('L_Hip')
    rhip_idx = joint_set['body']['joints_name'].index('R_Hip')
    pelvis = (joint_coord[lhip_idx,:] + joint_coord[rhip_idx,:]) * 0.5
    pelvis[2] = joint_coord[lhip_idx,2] * joint_coord[rhip_idx,2] # joint_valid
    pelvis = pelvis.reshape(1,3)
    
    # feet
    lfoot = feet_joint_coord[:3,:]
    rfoot = feet_joint_coord[3:,:]
    
    # hands
    lhand = ljoint_coord[[5,9,13,17], :]
    rhand = rjoint_coord[[5,9,13,17], :]
    joint_coord = np.concatenate((joint_coord, pelvis, lfoot, rfoot, lhand, rhand)).astype(np.float32)
    return joint_coord
args = parse_args()
cfg.set_args(args.gpu_ids, 'body')
cudnn.benchmark = True

# snapshot load
model_path = './snapshot_0.pth.tar'
assert osp.exists(model_path), 'Cannot find model at ' + model_path
print('Load checkpoint from {}'.format(model_path))
model = get_model('test')
model = DataParallel(model).cuda()
ckpt = torch.load(model_path)
model.load_state_dict(ckpt['network'], strict=False)
model.eval()


# prepare input image
transform = transforms.ToTensor()

img_path_dir = osp.join('..', 'data', 'MSCOCO', 'images')
annot_path = osp.join('..', 'data', 'MSCOCO', 'annotations')
db = COCO(osp.join(annot_path, 'coco_wholebody_train_v1.0.json'))
# mscoco joint set
joint_set = {'body': \
                    {'joint_num': 32, 
                    'joints_name': ('Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle', 'Pelvis', 'L_Big_toe', 'L_Small_toe', 'L_Heel', 'R_Big_toe', 'R_Small_toe', 'R_Heel', 'L_Index_1', 'L_Middle_1', 'L_Ring_1', 'L_Pinky_1', 'R_Index_1', 'R_Middle_1', 'R_Ring_1', 'R_Pinky_1'),
                    'flip_pairs': ( (1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16) , (18, 21), (19, 22), (20, 23), (24, 28), (25, 29) ,(26, 30), (27, 31) )
                    },\
            'hand': \
                    {'joint_num': 21,
                    'joints_name': ('Wrist', 'Thumb_1', 'Thumb_2', 'Thumb_3', 'Thumb_4', 'Index_1', 'Index_2', 'Index_3', 'Index_4', 'Middle_1', 'Middle_2', 'Middle_3', 'Middle_4', 'Ring_1', 'Ring_2', 'Ring_3', 'Ring_4', 'Pinky_1', 'Pinky_2', 'Pinky_3', 'Pinky_4'),
                    'flip_pairs': ()
                    },
            'face': \
                    {
                    'joint_to_flame': (-1, -1, -1, -1, -1, # no joints for neck, backheads, eyeballs
                                    17, 18, 19, 20, 21, # right eyebrow
                                    22, 23, 24, 25, 26, # left eyebrow
                                    27, 28, 29, 30, # nose
                                    31, 32, 33, 34, 35, # below nose
                                    36, 37, 38, 39, 40, 41, # right eye
                                    42, 43, 44, 45, 46, 47, # left eye
                                    48, # right lip
                                    49, 50, 51, 52, 53, # top lip
                                    54, # left lip
                                    55, 56, 57, 58, 59, # down lip
                                    60, 61, 62, 63, 64, 65, 66, 67, # inside of lip
                                    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 # face contour
                                    )
                    }
                }
count =0
coord_loss = CoordLoss()

for aid in db.anns.keys():
    ann = db.anns[aid]
    if ann['image_id'] != 1000:
        continue
    img = db.loadImgs(ann['image_id'])[0] 
    imgname = osp.join('train2017', img['file_name'])
    img_path = osp.join(img_path_dir, imgname)
    # body part
    if ann['iscrowd'] or (ann['num_keypoints'] == 0):
        continue
    
    # bbox
    bbox = process_bbox(ann['bbox'], img['width'], img['height']) 
    if bbox is None: continue
    img_shape= (img['height'],img['width'])
    # joint coordinates
    joint_img = np.array(ann['keypoints'], dtype=np.float32).reshape(-1,3)
    foot_joint_img = np.array(ann['foot_kpts'], dtype=np.float32).reshape(-1,3)
    ljoint_img = np.array(ann['lefthand_kpts'], dtype=np.float32).reshape(-1,3)
    rjoint_img = np.array(ann['righthand_kpts'], dtype=np.float32).reshape(-1,3)
    joint_img = add_joint(joint_img, foot_joint_img, ljoint_img, rjoint_img,joint_set)
    joint_valid = (joint_img[:,2].copy().reshape(-1,1) > 0).astype(np.float32)
    joint_img[:,2] = 0
    print("joint_img",joint_img)  

    original_img = load_img(img_path)
    img, img2bb_trans, bb2img_trans, rot, do_flip = augmentation(original_img, bbox, "val")
    trans_img = img.copy()
    img = transform(img.astype(np.float32))/255.
    img = img.cuda()[None,:,:,:]
    dummy_coord = np.zeros((joint_set['body']['joint_num'],3), dtype=np.float32)
    joint_img, joint_cam, joint_valid, joint_trunc = process_db_coord(joint_img, dummy_coord, joint_valid, do_flip, img_shape, joint_set['body']['flip_pairs'], img2bb_trans, rot, joint_set['body']['joints_name'], smpl.joints_name)
    if joint_trunc[1]*joint_trunc[2] ==0:
        print(imgname)
    #continue
    cv2.imwrite("AAA.jpg",draw_joints(trans_img.copy(),joint_img))       
    print("joint_img",joint_img)  
    start = time.time()

    # forward
    inputs = {'img': img}
    targets = {'joint_img': torch.from_numpy(joint_img).cuda()[None,:,:]}
    meta_info = {}
    with torch.no_grad():
        out = model(inputs, targets, meta_info, 'test')
    mesh = out['smpl_mesh_cam'].detach().cpu().numpy()[0]
    body_joint =  out['smpl_joint_proj'].detach().cpu().numpy()[0]
    
    print("body_joint", body_joint)
    print(time.time()-start)

    cv2.imwrite("bbb.jpg",draw_joints(trans_img.copy(),body_joint))
    # save mesh
    save_obj(mesh, smpl.face, 'output_body.obj')

    # render mesh
    vis_img = img.cpu().numpy()[0].transpose(1,2,0).copy() * 255
    rendered_img = render_mesh(vis_img, mesh, smpl.face, {'focal': cfg.focal, 'princpt': cfg.princpt})
    cv2.imwrite('render_cropped_img_body.jpg', rendered_img)

    vis_img = original_img.copy()
    focal = [cfg.focal[0] / cfg.input_img_shape[1] * bbox[2], cfg.focal[1] / cfg.input_img_shape[0] * bbox[3]]
    princpt = [cfg.princpt[0] / cfg.input_img_shape[1] * bbox[2] + bbox[0], cfg.princpt[1] / cfg.input_img_shape[0] * bbox[3] + bbox[1]]
    rendered_img = render_mesh(vis_img, mesh, smpl.face, {'focal': focal, 'princpt': princpt})
    cv2.imwrite('render_original_img_body.jpg', rendered_img)
    # save SMPL parameters
    smpl_pose = out['smpl_pose'].detach().cpu().numpy()[0]; smpl_shape = out['smpl_shape'].detach().cpu().numpy()[0]
    with open('smpl_param.json', 'w') as f:
        json.dump({'pose': smpl_pose.reshape(-1).tolist(), 'shape': smpl_shape.reshape(-1).tolist()}, f)
    count+=1
    from human_body_prior.body_model.body_model import BodyModel
    from body_visualizer.tools.vis_tools import render_smpl_params
    from body_visualizer.tools.vis_tools import imagearray2file

    bm_fname =  osp.join("","/Data/home/ducanh/human_body_prior/support_data/dowloads/models/smplx/neutral/model.npz")
    bm = BodyModel(bm_fname=bm_fname).to('cuda')
    images = render_smpl_params(bm, {'pose_body':out['smpl_pose'][:,3:66]}).reshape(1,1,1,400,400,3)
    img = imagearray2file(images)
    cv2.imwrite("aaa1.jpg",img[0])
    if count==1:
        break
