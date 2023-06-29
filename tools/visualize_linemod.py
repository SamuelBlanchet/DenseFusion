import _init_paths as _init_paths
import argparse
import os
import numpy as np
import yaml
from yaml.loader import SafeLoader
import copy
import torch
import torch.nn.parallel
import torch.utils.data
from datasets.linemod.dataset import PoseDataset as PoseDataset_linemod
from datasets.exo.dataset import PoseDataset as PoseDataset_exo
from lib.network import PoseNet, PoseRefineNet
from lib.loss import Loss
from lib.loss_refiner import Loss_refine
from lib.transformations import euler_matrix, quaternion_matrix, quaternion_from_matrix
#from lib.knn.__init__ import KNearestNeighbor
import trimesh
import cv2
import pyrender
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from transformations import concatenate_matrices

#### CONFIG
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', type=str, default = './datasets/linemod/Linemod_preprocessed', help='dataset root dir')
parser.add_argument('--model', type=str, default = 'trained_models/linemod/pose_model_7_0.012727497667224079.pth',  help='resume PoseNet model')
parser.add_argument('--refine_model', type=str, default = 'trained_models/linemod/pose_refine_model_253_0.006222824477970539.pth',  help='resume PoseRefineNet model')
parser.add_argument('--mode', type=str, default = 'eval',  help='Choose mode: eval or test')
opt = parser.parse_args()
####

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

def visualize_labels(img, mask, labelfile):
    if os.path.exists(labelfile):
        with open(labelfile, 'r') as fp:
            lines = fp.readlines()
            for line in lines:
                print("Label line: "+str(line))
                info = line.split()
        info = [float(i) for i in info]
        width, length = img.shape[:2]
        one = (int(info[3]*length),int(info[4]*width))
        two = (int(info[5]*length),int(info[6]*width))
        three = (int(info[7]*length),int(info[8]*width))
        four = (int(info[9]*length),int(info[10]*width))
        five = (int(info[11]*length),int(info[12]*width))
        six = (int(info[13]*length),int(info[14]*width))
        seven = (int(info[15]*length),int(info[16]*width))
        eight =  (int(info[17]*length),int(info[18]*width))

        cv2.line(img,one,two,(255,0,0),3)
        cv2.line(img,one,three,(255,0,0),3)
        cv2.line(img,two,four,(255,0,0),3)
        cv2.line(img,three,four,(255,0,0),3)
        cv2.line(img,one,five,(255,0,0),3)
        cv2.line(img,three,seven,(255,0,0),3)
        cv2.line(img,five,seven,(255,0,0),3)
        cv2.line(img,two,six,(255,0,0),3)
        cv2.line(img,four,eight,(255,0,0),3)
        cv2.line(img,six,eight,(255,0,0),3)
        cv2.line(img,five,six,(255,0,0),3)
        cv2.line(img,seven,eight,(255,0,0),3)
    else:
        print("DID NOTHING")
    return img

def render_object_pose(meshname, matrix):
    # Uncolored mesh
    #part_trimesh = trimesh.load("../datasets/exo/Exo_preprocessed/objects/259.stl")
    reservoir_1 = trimesh.load(meshname)       # [-0.8830567, -0.5400259, -0.7] pour environ 0
    reservoir_1.visual.vertex_colors = np.full(reservoir_1.vertices.shape, [255, 0, 0])
    reservoir_2 = trimesh.load(meshname)       # [-0.8830567, -0.5400259, -0.7] pour environ 0
    reservoir_2.visual.vertex_colors = np.full(reservoir_2.vertices.shape, [0, 255, 0])

    # Initial transformation to have the mesh at it's origin
    init_pose = np.array(np.eye(4))
    example_pose = np.array(np.eye(4))
    init_pose[:3,3] = [-88.30567, -54.00259, 0]
    example_pose[:3,3] = [-88.30567, -54.00259, -200]
    #print("Init pose: "+str(init_pose))

    # Colored mesh
    #part_trimesh = trimesh.load("../datasets/exo/Exo_preprocessed/objects/reservoir_transparent_A_colored.ply")

    # Sphere from example
    #sphere = trimesh.creation.icosphere(subdivisions=4, radius=0.8)
    #sphere.vertices+=1e-2*np.random.randn(*sphere.vertices.shape)

    #mesh = pyrender.Mesh.from_trimesh(sphere, smooth=False)
    mesh_reservoir_1 = pyrender.Mesh.from_trimesh(reservoir_1, smooth=False)
    mesh_reservoir_2 = pyrender.Mesh.from_trimesh(reservoir_2, smooth=False)

    # compose scene
    scene = pyrender.Scene(ambient_light=[.1, .1, .3], bg_color=[100, 100, 100])
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    # cam_K: [18.1476, 0.0, 20.955, 0.0, 18.1476, 15.2908, 0.0, 0.0, 1.0]
    persp_camera = pyrender.IntrinsicsCamera(fx=18.1476, fy=18.1476, cx=20.955, cy=15.2908)
    light = pyrender.DirectionalLight(color=[1,1,1], intensity=2e3)

    pred = np.array(np.eye(4))
    pred[:3,:] = matrix
    pred[:3,3] = pred[:3,3]

    mesh_pose = concatenate_matrices(init_pose, pred)

    scene.add(mesh_reservoir_1, pose=mesh_pose)
    scene.add(mesh_reservoir_2, pose=init_pose)
    scene.add(light, pose=np.eye(4))

    #c = 2**-0.5
    #scene.add(camera, pose=[[ 1,  0,  0,  0],
    #                        [ 0,  c, -c, -2],
    #                        [ 0,  c,  c,  2],
    #                        [ 0,  0,  0,  1]])
    cam_matrix=np.array(np.eye(4))
    #matrix[:,3]=np.array([0, 0, 0, 1])     # Positions x, y, z  ,         [-0.8830567, -0.5400259, -0.7]
    #rotation = R.from_euler('xyz', [0, 0, 0], degrees=True)
    #matrix[0:3,0:3]=rotation.as_matrix()
    scene.add(camera, pose=cam_matrix)
    #scene.add(persp_camera, pose=matrix)

    # render scene
    pyrender.Viewer(scene)
    r = pyrender.OffscreenRenderer(1024, 1024)
    color, _ = r.render(scene)
    plt.imsave("test.png", color)
    return

if __name__ == '__main__':
    opt.num_objects = 13
    opt.num_points = 500
    opt.iteration = 4
    opt.objlist = [1, 2, 3]
    bs = 1
    dataset_config_dir = 'datasets/linemod/dataset_config'

    vis_bbox = True

    # For visualizing labels
    intrinsic_file = open( f'{opt.dataset_root}/data/0{opt.objlist[0]}/info.yml')
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
    estimator = PoseNet(num_points = opt.num_points, num_obj = opt.num_objects)
    estimator = estimator.cuda()
    refiner = PoseRefineNet(num_points = opt.num_points, num_obj = opt.num_objects)
    refiner = refiner.cuda()
    estimator.load_state_dict(torch.load(opt.model))
    refiner.load_state_dict(torch.load(opt.refine_model))
    estimator.eval()
    refiner.eval()

    testdataset = PoseDataset_linemod('test', opt.num_points, True, opt.dataset_root, 0.0, True)
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

    opt.success_count = [0 for i in range(opt.num_objects)]
    opt.num_count = [0 for i in range(opt.num_objects)]

    diameter = []
    real_poses = []
    meta_file = open('{0}/models_info.yml'.format(dataset_config_dir), 'r')
    meta = yaml.load(meta_file, SafeLoader)

    for obj in opt.objlist:
        diameter.append((meta[obj]['diameter'] / 1000.0) * 0.1)

    for i, data in enumerate(testdataloader, 0):
        image = cv2.imread(ori_image_path[i])
        mask = cv2.imread(ori_mask_path[i])

        points, choose, img, target, model_points, idx = data
        print(model_points)

        scene_number = ori_image_path[i].rsplit('/', 1)[-1][:-4]

        points, choose, img, target, model_points, idx = (points).cuda(), \
                                                        (choose).cuda(), \
                                                        (img).cuda(), \
                                                        (target).cuda(), \
                                                        (model_points).cuda(), \
                                                        (idx).cuda()
        
        #meshname = f'datasets/exo/Exo_preprocessed/models/obj_0{opt.objlist[0]}.ply'
        meshname = ori_meshes_path[idx.item()]
        mesh = trimesh.load(meshname)
        vertices = np.c_[np.array(mesh.vertices), np.ones((len(mesh.vertices), 1))].transpose()
        corners3D = get_3D_corners(mesh, 'from_file', models_infos[idx.item()+1])
        coord3D = np.array([[0,  0.05,     0,     0],
                            [0,    0,   0.05,     0],
                            [0,    0,     0,   0.05],
                            [1,    1,     1,     1]])

        if opt.mode == "eval":
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
                if opt.mode == "eval":
                    pred_r, pred_t = refiner(new_points, emb, idx)
                    pred_r = pred_r.view(1, 1, -1)
                    pred_r = pred_r / (torch.norm(pred_r, dim=2).view(1, 1, 1))
                    my_r_2 = pred_r.view(-1).cpu().data.numpy()
                    my_t_2 = pred_t.view(-1).cpu().data.numpy()
                    my_mat_2 = quaternion_matrix(my_r_2)
                    my_mat_2[0:3, 3] = my_t_2

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
                        ground_truth = gt_list[idx.item()+1][int(scene_number)][0]
                        gt_rotation = ground_truth['cam_R_m2c']
                        gt_translation = ground_truth['cam_t_m2c']
                        print("GT rotation: "+str(gt_rotation))
                        print("GT translation: "+str(gt_translation))       # En metres
                    except:
                        print("GT not found lol")


        elif opt.mode == 'test':
            ground_truth = gt_list[idx.item()+1][int(scene_number)][0]
            my_R = ground_truth['cam_R_m2c']
            my_T = ground_truth['cam_t_m2c']
            my_t = my_T
            my_r = np.array(my_R)
            my_R = my_r.reshape(3, 3)
            my_T = np.reshape(my_t,(len(my_t), 1))
            my_T = [i / 1000 for i in my_T]


        print("Rotation: "+str(my_R))
        print("Translation: "+str(my_T))

        model_points = model_points[0].cpu().detach().numpy()
        my_r = quaternion_matrix(my_r)[:3, :3]
        pred = np.dot(model_points, my_r.T) + my_t
        target = target[0].cpu().detach().numpy()

        dis = np.mean(np.linalg.norm(pred - target, axis=1))

        # Load object Diameter
        #print("Image path: "+str(ori_image_path[i]))
        #print("Mask path: "+str(ori_mask_path[i]))
        #if dis < diameter[idx[0].item()]:
        #    opt.success_count[idx[0].item()] += 1
        #    print('No.{0} Pass! Distance: {1}, Diameter: {2}'.format(i, dis, diameter))
        #else:
        #    print('No.{0} NOT Pass! Distance: {1}'.format(i, dis))
        #opt.num_count[idx[0].item()] += 1

        if (vis_bbox == True):
            Rt_pred = np.concatenate((my_R, my_T), axis=1)

            proj_2d_pred = compute_projection(vertices, Rt_pred, intrinsic_calibration)
            proj_corners_pred = np.transpose(compute_projection(corners3D, Rt_pred, intrinsic_calibration))
            proj_coord_pred = np.transpose(compute_projection(coord3D, Rt_pred, intrinsic_calibration))

            # 3D bounding box visualization
            for edge in edges_corners:
                center_coordinates1 = (int(proj_corners_pred[edge[0], 0]), int(proj_corners_pred[edge[0], 1]))
                center_coordinates2 = (int(proj_corners_pred[edge[1], 0]), int(proj_corners_pred[edge[1], 1]))
                cv2.line(image, center_coordinates1, center_coordinates2, (0, 0, 255), 2)

            # Coordinate system visualization
            i = 0
            for edge in coord_sys_edges:
                if i == 0:
                    color = (0, 0, 255)
                elif i == 1:
                    color = (0, 255, 0)
                else:
                    color = (255, 0, 0)
                center_coordinates1 = (int(proj_coord_pred[edge[0], 0]), int(proj_coord_pred[edge[0], 1]))
                center_coordinates2 = (int(proj_coord_pred[edge[1], 0]), int(proj_coord_pred[edge[1], 1]))
                cv2.line(image, center_coordinates1, center_coordinates2, color, 2)
                i+=1

            cv2.addWeighted(mask, 0.4, image, 0.6, 0, image)
            #image = visualize_labels(image, mask, ori_label_path[i])       # Fait rien pour le moment, et a 18 ca chise
        cv2.imshow('Scene with prediction', image)
        cv2.waitKey(0)

    for i in range(opt.num_objects):
        print('Object {0} success rate: {1}'.format(opt.objlist[i], float(opt.success_count[i]) / opt.num_count[i]))
    print('ALL success rate: {0}'.format(float(sum(opt.success_count)) / sum(opt.num_count)))