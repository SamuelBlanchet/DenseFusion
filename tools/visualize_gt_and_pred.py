import _init_paths as _init_paths
import argparse
import os
import numpy as np
import yaml
from yaml.loader import SafeLoader
import copy
import torch
#import torch.nn as nn
import torch.nn.parallel
#import torch.backends.cudnn as cudnn
#import torch.optim as optim
import torch.utils.data
#import torchvision.datasets as dset
#import torchvision.transforms as transforms
#import torchvision.utils as vutils
from datasets.linemod.dataset import PoseDataset as PoseDataset_linemod
from datasets.exo.dataset import PoseDataset as PoseDataset_exo
from datasets.reservoir_transparent_a.dataset import PoseDataset as PoseDataset_exo_reservoir
from datasets.part_259.dataset import PoseDataset as PoseDataset_exo_259
from lib.network import PoseNet, PoseRefineNet
from lib.loss import Loss
from lib.loss_refiner import Loss_refine
from lib.transformations import euler_matrix, quaternion_matrix, quaternion_from_matrix
from lib.knn.__init__ import KNearestNeighbor
import trimesh
import cv2
import pyrender
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from transformations import concatenate_matrices

#### CONFIG
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default = 'exo', help='dataset name: exo, linemod, exo_259, exo_reservoir')
parser.add_argument('--model', type=str, default = '',  help='PoseNet model')
parser.add_argument('--refine_model', type=str, default = '',  help='PoseRefineNet model')
parser.add_argument('--mode', type=str, default = 'show_both',  help='Choose mode: eval or show_gt')
opt = parser.parse_args()
####

exo = {'num_obj': 7, 'dataset_root': './datasets/exo/Exo_preprocessed', 'dataset_config_dir':'datasets/exo/dataset_config'}
linemod = {'num_obj': 13, 'dataset_root': './datasets/linemod/Linemod_preprocessed', 'dataset_config_dir':'datasets/linemod/dataset_config'}
exo_reservoir = {'num_obj': 1, 'dataset_root': './datasets/reservoir_transparent_a/Reservoir_preprocessed', 'dataset_config_dir':'datasets/reservoir_transparent_a/dataset_config'}
exo_259 = {'num_obj': 1, 'dataset_root': './datasets/part_259/259_preprocessed', 'dataset_config_dir':'datasets/part_259/dataset_config'}

def get_camera_intrinsic(u0, v0, fx, fy):
    return np.array([[fx, 0.0, u0], [0.0, fy, v0], [0.0, 0.0, 1.0]])

def get_3D_corners(mesh, method, obj_infos):
    if method == 'from_file':
        min_x = obj_infos['min_x']*0.001
        min_y = obj_infos['min_y']*0.001
        min_z = obj_infos['min_z']*0.001
        max_x = (obj_infos['min_x']+obj_infos['size_x'])*0.001
        max_y = (obj_infos['min_y']+obj_infos['size_y'])*0.001
        max_z = (obj_infos['min_z']+obj_infos['size_z'])*0.001
    else:
        #Tform = mesh.apply_obb()
        points = mesh.bounding_box.vertices
        #center = mesh.centroid
        min_x = np.min(points[:,0])*0.01
        min_y = np.min(points[:,1])*0.01
        min_z = np.min(points[:,2])*0.01
        max_x = np.max(points[:,0])*0.01
        max_y = np.max(points[:,1])*0.01
        max_z = np.max(points[:,2])*0.01
    
    corners = np.array([[min_x, min_y, min_z], [min_x, min_y, max_z], [min_x, max_y, min_z],
                        [min_x, max_y, max_z], [max_x, min_y, min_z], [max_x, min_y, max_z],
                        [max_x, max_y, min_z], [max_x, max_y, max_z]])
    corners = np.concatenate((np.transpose(corners), np.ones((1,8)) ), axis=0)
    return corners

def compute_projection(points_3D, transformation, internal_calibration):
    projections_2d = np.zeros((2, points_3D.shape[1]), dtype='float32')
    camera_projection = (internal_calibration.dot(transformation)).dot(points_3D)
    projections_2d[0, :] = camera_projection[0, :]/camera_projection[2, :]
    projections_2d[1, :] = camera_projection[1, :]/camera_projection[2, :]
    return projections_2d

if __name__ == '__main__':
    opt.num_points = 500
    opt.iteration = 4
    bs = 1
    if opt.dataset_name == 'exo':
        param = exo
        testdataset = PoseDataset_exo(opt.mode, opt.num_points, True, param['dataset_root'], 0.0, True)
        obj_list = [1, 2, 3, 4, 5, 6, 7]
    elif opt.dataset_name == 'linemod':
        param = linemod
        testdataset = PoseDataset_linemod(opt.mode, opt.num_points, True, param['dataset_root'], 0.0, True)
        obj_list = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
    elif opt.dataset_name == 'exo_259':
        param = exo_259
        testdataset = PoseDataset_exo_259(opt.mode, opt.num_points, True, param['dataset_root'], 0.0, True)
        obj_list = [1]
    elif opt.dataset_name == 'exo_reservoir':
        param = exo_reservoir
        testdataset = PoseDataset_exo_reservoir(opt.mode, opt.num_points, True, param['dataset_root'], 0.0, True)
        obj_list = [1]
    
    # Create object list***
    #obj_list = []
    #for obj_id in range(1, param['num_obj']+1):
    #    obj_list.append(obj_id)
    #if 
    print('Object list: '+str(obj_list))
    
    dataset_config_dir = param['dataset_config_dir']

    vis_bbox = True

    # For visualizing labels
    intrinsic_file = open(param['dataset_root']+f'/data/0{obj_list[0]}/info.yml')
    edges_corners = [[0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3], [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]]
    coord_sys_edges = [[0, 1],[0, 2],[0, 3]]

    # Load Camera intrinsic array and depth scale
    cam_data = yaml.load(intrinsic_file, Loader=SafeLoader)
    first_scene = list(cam_data.keys())[0]
    # cam_K = [569.6790, 0.0, 240, 0.0, 554.2574, 320, 0.0, 0.0, 1.0] = [[fx, 0, x0], [0, fy, y0], [0, 0, 1]]
    fx = cam_data[first_scene]['cam_K'][0]
    fy = cam_data[first_scene]['cam_K'][4]
    cx = cam_data[first_scene]['cam_K'][2]
    cy = cam_data[first_scene]['cam_K'][5]

    intrinsic_calibration = get_camera_intrinsic(cx,cy,fx,fy)
    if opt.mode == "eval" or opt.mode == "show_both":
        estimator = PoseNet(num_points = opt.num_points, num_obj = param['num_obj'])
        estimator = estimator.cuda()
        estimator.load_state_dict(torch.load(opt.model))
        estimator.eval()
        if opt.refine_model != '':
            refiner = PoseRefineNet(num_points = opt.num_points, num_obj = param['num_obj'])
            refiner = refiner.cuda()
            refiner.load_state_dict(torch.load(opt.refine_model))
            refiner.eval()

    testdataloader = torch.utils.data.DataLoader(testdataset, batch_size=1, shuffle=False, num_workers=10)

    sym_list = testdataset.get_sym_list()
    num_points_mesh = testdataset.get_num_points_mesh()
    ori_image_path = testdataset.get_img()
    ori_mask_path = testdataset.get_mask()
    ori_meshes_path = testdataset.get_mesh()
    ori_label_path = testdataset.get_ssp_label()
    gt_list = testdataset.meta
    models_infos = testdataset.infos
    criterion = Loss(num_points_mesh, sym_list)
    criterion_refine = Loss_refine(num_points_mesh, sym_list)

    opt.success_count = [0 for i in range(param['num_obj'])]
    opt.num_count = [0 for i in range(param['num_obj'])]

    diameter = []
    real_poses = []
    meta_file = open('{0}/models_info.yml'.format(dataset_config_dir), 'r')
    meta = yaml.load(meta_file, SafeLoader)

    for obj in obj_list:
        diameter.append((meta[obj]['diameter'] / 1000.0) * 0.1)

    print("Lenght of the testing dataset: "+str(len(testdataloader)))

    for i, data in enumerate(testdataloader, 0):
        image = cv2.imread(ori_image_path[i])
        mask = cv2.imread(ori_mask_path[i])
        points, choose, img, target, model_points, idx = data

        scene_number = ori_image_path[i].rsplit('/', 1)[-1][:-4]

        points, choose, img, target, model_points, idx = (points).cuda(), \
                                                        (choose).cuda(), \
                                                        (img).cuda(), \
                                                        (target).cuda(), \
                                                        (model_points).cuda(), \
                                                        (idx).cuda()

        meshname = ori_meshes_path[idx.item()]
        mesh = trimesh.load(meshname)
        vertices = np.c_[np.array(mesh.vertices), np.ones((len(mesh.vertices), 1))].transpose()
        corners3D = get_3D_corners(mesh, 'from_file', models_infos[idx.item()+1])
        coord3D = np.array([[0,  0.05,     0,     0],
                            [0,    0,   0.05,     0],
                            [0,    0,     0,   0.05],
                            [1,    1,     1,     1]])

        if opt.mode == "eval" or opt.mode == 'show_both':
            pred_r, pred_t, pred_c, emb = estimator(img, points, choose, idx)
            pred_r = pred_r / torch.norm(pred_r, dim=2).view(1, opt.num_points, 1)
            pred_c = pred_c.view(bs, opt.num_points)
            how_max, which_max = torch.max(pred_c, 1)
            pred_t = pred_t.view(bs * opt.num_points, 1, 3)
            my_r = pred_r[0][which_max[0]].view(-1).cpu().data.numpy()
            my_t = (points.view(bs * opt.num_points, 1, 3) + pred_t)[which_max[0]].view(-1).cpu().data.numpy()
            my_pred = np.append(my_r, my_t)

            for ite in range(0, opt.iteration):
                T = (torch.from_numpy(my_t.astype(np.float32))).cuda().view(1, 3).repeat(opt.num_points, 1).contiguous().view(1, opt.num_points, 3)
                my_mat = quaternion_matrix(my_r)
                R = (torch.from_numpy(my_mat[:3, :3].astype(np.float32))).cuda().view(1, 3, 3)
                my_mat[0:3, 3] = my_t

                new_points = torch.bmm((points - T), R).contiguous()
                if opt.mode == "eval" or opt.mode == 'show_both':
                    if opt.refine_model != '':
                        print("Refining with model "+str(opt.refine_model))
                        pred_r, pred_t = refiner(new_points, emb, idx)
                        pred_r = pred_r.view(1, 1, -1)
                        pred_r = pred_r / (torch.norm(pred_r, dim=2).view(1, 1, 1))     # Pourquoi 1 ici et pas opt.num_points
                        my_r_2 = pred_r.view(-1).cpu().data.numpy()
                        my_t_2 = pred_t.view(-1).cpu().data.numpy()
                        my_mat_2 = quaternion_matrix(my_r_2)
                        my_mat_2[0:3, 3] = my_t_2
                    else:
                        my_mat_2 = quaternion_matrix(my_r)
                        my_mat_2[0:3, 3] = my_t
            
                    my_mat_final = np.dot(my_mat, my_mat_2)
                    my_r_final = copy.deepcopy(my_mat_final)
                    my_r_final[0:3, 3] = 0
                    my_R = my_mat_final[0:3, 0:3]
                    my_r_final = quaternion_from_matrix(my_r_final, True)
                    my_t_final = np.array([my_mat_final[0][3], my_mat_final[1][3], my_mat_final[2][3]])
                    my_t_test = np.array([my_mat[0][3], my_mat[1][3], my_mat[2][3]])

                    my_pred_final = np.append(my_r_final, my_t_final)
                    my_r = my_r_final
                    my_t = my_t_final
                    my_R = np.array(my_R).reshape(3, 3)
                    my_T = np.reshape(my_t,(len(my_t), 1))

                    try:
                        if idx>=3:
                            ground_truth = gt_list[idx.item()+2][int(scene_number)][0]
                        elif idx>=7:
                            ground_truth = gt_list[idx.item()+3][int(scene_number)][0]
                        else:
                            ground_truth = gt_list[idx.item()+1][int(scene_number)][0]
                    except:
                        print("No ground truth found...")
                        ground_truth = {'cam_R_m2c': [1, 0, 0, 0, 1, 0, 0, 0, 1],
                                        'cam_t_m2c': [0, 0, 1000], 'obj_bb': [244, 150, 44, 58], 'obj_id': 1}
                    gt_rotation = ground_truth['cam_R_m2c']
                    gt_translation = ground_truth['cam_t_m2c']
            print("GT rotation: "+str(gt_rotation))
            print("GT translation: "+str(gt_translation))

        if opt.mode == "show_gt" or opt.mode == 'show_both':
            try:
                if idx>=3:
                    ground_truth = gt_list[idx.item()+2][int(scene_number)][0]
                elif idx>=7:
                    ground_truth = gt_list[idx.item()+3][int(scene_number)][0]
                else:
                    ground_truth = gt_list[idx.item()+1][int(scene_number)][0]
            except:
                print("what")
                #ground_truth = gt_list[idx.item()+1][int(scene_number)][0]
                ground_truth = {'cam_R_m2c': [1, 0, 0, 0, 1, 0, 0, 0, 1],
                                'cam_t_m2c': [0, 0, 1000], 'obj_bb': [244, 150, 44, 58], 'obj_id': 1}
                print("No ground truth found...")
            my_gt_R = ground_truth['cam_R_m2c']
            my_gt_T = ground_truth['cam_t_m2c']
            my_gt_t = my_gt_T
            my_gt_r = np.array(my_gt_R)
            my_gt_R = my_gt_r.reshape(3, 3)
            my_gt_T = np.reshape(my_gt_t,(len(my_gt_t), 1))
            my_gt_T = [i / 1000 for i in my_gt_T]

        print("Predicted rotation: "+str(my_R))
        print("Predicted translation: "+str(my_T))

        model_points = model_points[0].cpu().detach().numpy()
        my_r = quaternion_matrix(my_r)[:3, :3]
        pred = np.dot(model_points, my_r.T) + my_t
        target = target[0].cpu().detach().numpy()

        dis = np.mean(np.linalg.norm(pred - target, axis=1))

        # Load object Diameter
        print("Image path: "+str(ori_image_path[i])+"\n")
        #print("Mask path: "+str(ori_mask_path[i]))
        if opt.mode == "eval" or opt.mode == "show_both":
            if dis < diameter[idx[0].item()]:
                opt.success_count[idx[0].item()] += 1
                #print('No.{0} Pass! Distance: {1}, Diameter: {2}'.format(i, dis, diameter))
                print('No.{0} Pass! Distance: {1}'.format(i, dis))
            else:
                print('No.{0} NOT Pass! Distance: {1}'.format(i, dis))
            opt.num_count[idx[0].item()] += 1

        if (vis_bbox == True):
            Rt_pred = np.concatenate((my_R, my_T), axis=1)

            proj_2d_pred = compute_projection(vertices, Rt_pred, intrinsic_calibration)
            proj_corners_pred = np.transpose(compute_projection(corners3D, Rt_pred, intrinsic_calibration))
            proj_coord_pred = np.transpose(compute_projection(coord3D, Rt_pred, intrinsic_calibration))

            # Prediction visualization
            for edge in edges_corners:
                center_coordinates1 = (int(proj_corners_pred[edge[0], 0]), int(proj_corners_pred[edge[0], 1]))
                center_coordinates2 = (int(proj_corners_pred[edge[1], 0]), int(proj_corners_pred[edge[1], 1]))
                cv2.line(image, center_coordinates1, center_coordinates2, (0, 0, 255), 2)

            # Coordinate system visualization
            #i = 0
            #for edge in coord_sys_edges:
            #    if i == 0:
            #        color = (0, 0, 255)
            #    elif i == 1:
            #        color = (0, 255, 0)
            #    else:
            #        color = (255, 0, 0)
            #    center_coordinates1 = (int(proj_coord_pred[edge[0], 0]), int(proj_coord_pred[edge[0], 1]))
            #    center_coordinates2 = (int(proj_coord_pred[edge[1], 0]), int(proj_coord_pred[edge[1], 1]))
            #    cv2.line(image, center_coordinates1, center_coordinates2, color, 2)
            #    i+=1

            # Groundtruth visualization
            if opt.mode == "show_gt" or opt.mode == 'show_both':
                Rt_pred = np.concatenate((my_gt_R, my_gt_T), axis=1)
                proj_2d_pred = compute_projection(vertices, Rt_pred, intrinsic_calibration)
                proj_corners_pred = np.transpose(compute_projection(corners3D, Rt_pred, intrinsic_calibration))
                proj_coord_pred = np.transpose(compute_projection(coord3D, Rt_pred, intrinsic_calibration))
                for edge in edges_corners:
                    center_coordinates1 = (int(proj_corners_pred[edge[0], 0]), int(proj_corners_pred[edge[0], 1]))
                    center_coordinates2 = (int(proj_corners_pred[edge[1], 0]), int(proj_corners_pred[edge[1], 1]))
                    cv2.line(image, center_coordinates1, center_coordinates2, (0, 255, 0), 2)

                #cv2.addWeighted(mask, 0.4, image, 0.6, 0, image)       # View GT mask

        cv2.imshow('Scene with prediction', image)
        cv2.waitKey(0)
    if opt.mode == "eval" or opt.mode == "show_both":
        for i in range(param['num_obj']):
            print('Object {0} success rate: {1}'.format(obj_list[i], float(opt.success_count[i]) / opt.num_count[i]))
        print('ALL success rate: {0}'.format(float(sum(opt.success_count)) / sum(opt.num_count)))
