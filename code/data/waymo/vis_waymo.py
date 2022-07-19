import sys,os,shutil
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle
SPLIT = '3dop'
DATA_PATH = 'data/waymo2kitti/validation/segment-1863454917318776530_1040_000_1060_000_with_camera_labels/'    # ../../data/kitti/training/calib/
PRED_PATH = '/home/ctl/wl/test/output4/data/'
from image_utils import compute_box_3d, project_to_image, alpha2rot_y, draw_box_centers
from image_utils import draw_box_3d, unproject_2d_to_3d, draw_box_2d, plot_points_on_image,rgba

'''
#Values    Name      Description
----------------------------------------------------------------------------
   1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                     'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                     'Misc' or 'DontCare'
   1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                     truncated refers to the object leaving image boundaries
   1    occluded     Integer (0,1,2,3) indicating occlusion state:
                     0 = fully visible, 1 = partly occluded
                     2 = largely occluded, 3 = unknown
   1    alpha        Observation angle of object, ranging [-pi..pi]
   4    bbox         2D bounding box of object in the image (0-based index):
                     contains left, top, right, bottom pixel coordinates
   3    dimensions   3D object dimensions: height, width, length (in meters)
   3    location     3D object location x,y,z in camera coordinates (in meters)
   1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
   1    score        Only for results: Float, indicating confidence in
                     detection, needed for p/r curves, higher is better.
'''
cats = ['Car', 'Pedestrian', 'Cyclist']
cat_ids = {cat: i + 1 for i, cat in enumerate(cats)}

def read_clib(calib_path):
    f = open(calib_path, 'r')
    for i, line in enumerate(f):
        if i == 2:
            calib = np.array(line[:-1].split(' ')[1:], dtype=np.float32)
            calib = calib.reshape(3, 4)
            return calib


# image_set_path = DATA_PATH + 'ImageSets_{}/'.format(SPLnt,IT)
ann_dir = DATA_PATH + 'label_0/'
calib_dir = DATA_PATH + 'calib/'
anno_list = os.listdir(ann_dir) #['000001.txt']
depth_list = []
for file_name in anno_list:
    #import ipdb;ipdb.set_trace()
    calib_path = calib_dir + file_name
    calib = read_clib(calib_path)
    image_info = {'file_name': '{}.png'.format(file_name.split('.')[0]),
                'id': int(file_name.split('.')[0]),
                'calib': calib.tolist()}
    # if image_info['file_name'] != '000213.png': continue
    image = cv2.imread(DATA_PATH + 'image_0/' + image_info['file_name'])

    ann_path = ann_dir + file_name
    pred_ann_path = PRED_PATH + file_name



    anns = open(ann_path, 'r')
    boxes_for_pro = []
    for ann_ind, txt in enumerate(anns):
        tmp = txt[:-1].split(' ')
        cat_id = cat_ids[tmp[0]]
        truncated = int(float(tmp[1]))
        occluded = int(tmp[2])
        alpha = float(tmp[3])
        bbox = [float(tmp[4]), float(tmp[5]), float(tmp[6]), float(tmp[7])]
        dim = [float(tmp[8]), float(tmp[9]), float(tmp[10])]
        location = [-float(tmp[12]), -float(tmp[13]), float(tmp[11])]
        rotation_y = float(tmp[14])
        # projected 3D center
        
        if (bbox[2] - bbox[0]) == 0 and (bbox[3] - bbox[1]) == 0:
            continue
        boxes_for_pro.append(bbox + [location[-1]])
        box_3d = compute_box_3d(dim, location, rotation_y)
        print(box_3d[:,0].max())
        
        # draw image
        box_2d = project_to_image(box_3d, calib)
        x_min = box_2d[:, 0].min(0)
        x_max = box_2d[:, 0].max(0)
        y_min = box_2d[:, 1].min(0)
        y_max = box_2d[:, 1].max(0)
        print('box_2d', x_min,x_max,y_min,y_max,bbox)
        image = draw_box_3d(image, box_2d)
        image = draw_box_2d(image, bbox)

    projected_path = DATA_PATH + 'projected_points_0/' + file_name.replace('.txt', '.npy')
    cps_all = np.load(projected_path)
    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plot_points_on_image(cps_all, rgb_img, rgba, boxes_for_pro, 'vis_waymo/'+file_name.split('.')[0]+'.png')




