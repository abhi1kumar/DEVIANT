"""
    Sample Run:
    python plot/plot_colored_point_cloud.py
    
    Plots colored point cloud. (Requires installing open3d)
"""
import os, sys
sys.path.append(os.getcwd())

import numpy as np
np.set_printoptions   (precision= 4, suppress= True)

import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt
from PIL import Image

def get_calib_from_file(calib_file):
    with open(calib_file) as f:
        lines = f.readlines()

    obj = lines[2].strip().split(' ')[1:]
    P2 = np.array(obj, dtype=np.float32)
    obj = lines[3].strip().split(' ')[1:]
    P3 = np.array(obj, dtype=np.float32)
    obj = lines[4].strip().split(' ')[1:]
    R0 = np.array(obj, dtype=np.float32)
    obj = lines[5].strip().split(' ')[1:]
    Tr_velo_to_cam = np.array(obj, dtype=np.float32)

    return {'P2': P2.reshape(3, 4),
            'P3': P3.reshape(3, 4),
            'R0': R0.reshape(3, 3),
            'Tr_velo2cam': Tr_velo_to_cam.reshape(3, 4)}


def projective_transform(camera_mat, points):
    proj_points   = camera_mat @ points.transpose() # 4 x N
    proj_points[:2] /= (proj_points[2] + 1e-3)
    return proj_points.transpose()

def visualize_and_save(vis, colored_point_cloud, cnt):
    # Visualize Point Cloud
    vis.add_geometry(colored_point_cloud)

    # View Control
    # https://github.com/isl-org/Open3D/issues/6121#issue-1686487894
    view_ctl: o3d.visualization.ViewControl = vis.get_view_control()
    view_ctl.set_up((0, 0, 1))  # set the negative direction of the y-axis as the up direction
    view_ctl.set_front((-1, 0.0, 0))  # set the positive direction of the x-axis toward you
    view_ctl.set_lookat((50-0.1*cnt, 0+0.025*cnt, 0))  # set the original point as the center point of the window
    view_ctl.set_zoom(0.39)

    # vis.run()
    # vis.update_geometry(colored_point_cloud)
    # vis.poll_events()
    # vis.update_renderer()
    # vis.capture_screen_image("images/sample_render.png")

    img = np.array(vis.capture_screen_float_buffer(True))[350:850, 320:1850]
    plt.figure(figsize=(10, 6), dpi=200)
    plt.imshow(img)
    # plt.gca().get_xaxis().set_ticks([])
    # plt.gca().get_yaxis().set_ticks([])
    plt.axis('off')
    folder_name = "images/depth_translation/"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    file_path = folder_name + str(cnt).zfill(6) + ".png"
    plt.savefig(file_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()

    # Remove previous geometry
    vis.remove_geometry(colored_point_cloud)

point_cloud_path = 'data/kitti/training/velodyne/000008.bin'
rgb_image_path   = 'data/kitti/training/image_2/000008.png'
calib_path       = 'data/kitti/training/label_2/000008.txt'

calib = get_calib_from_file(calib_path)
P2_t  = calib['P2']  # 3 x 4
R0_t  = calib['R0']  # 3 x 3
V2C_t = calib['Tr_velo2cam']  # 3 x 4

P2 = np.eye(4)
P2[:3] = P2_t
R0 = np.eye(4)
R0[:3, :3] = R0_t
V2C = np.eye(4)
V2C[:3] = V2C_t

camera_mat = P2 @ R0 @ V2C
print(camera_mat)

# Load the RGB image
rgb_image = np.array(Image.open(rgb_image_path))

# Load the KITTI point cloud data (.bin format)
points = np.fromfile(point_cloud_path, dtype=np.float32).reshape(-1, 4)  # Assumes XYZI format (4 channels)

pc_points = np.ones((points.shape[0], 4))
pc_points[:, :3] = points[:, :3]
pixel_locations = projective_transform(camera_mat, pc_points).astype(np.int64)
image_height, image_width, _ = rgb_image.shape

# Crop the point cloud
th = 20
valid_index = np.logical_and(np.logical_and(np.logical_and(
    np.logical_and(pixel_locations[:, 2] >= -0.5, pixel_locations[:, 0] >= -th),
    pixel_locations[:, 0] <= image_width-1+th), pixel_locations[:, 1] >= -th), pixel_locations[:, 1] <= image_height-1+th)
points                = points[valid_index]
pixel_locations       = pixel_locations[valid_index]
# Clamp pixel locations to valid image coordinates
pixel_locations[:, 0] = np.clip(pixel_locations[:, 0], 0, image_width - 1)
pixel_locations[:, 1] = np.clip(pixel_locations[:, 1], 0, image_height - 1)
# Extract RGB values from the image corresponding to the point cloud
colors = rgb_image[pixel_locations[:, 1], pixel_locations[:, 0], :]

# Create a new point cloud with XYZ and color attributes
colored_point_cloud = o3d.geometry.PointCloud()
colored_point_cloud.points = o3d.utility.Vector3dVector(points[:, :3])  # XYZ coordinates
colored_point_cloud.colors = o3d.utility.Vector3dVector(colors / 255.0)  # Normalize RGB values to [0, 1]
# colored_point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=3))
# mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(colored_point_cloud, depth=11)


# Visualize the colored point cloud
# Option 1
# See https://github.com/isl-org/Open3D/issues/6121
vis = o3d.visualization.Visualizer()
vis.create_window(visible=False)

option: o3d.visualization.RenderOption = vis.get_render_option()
option.show_coordinate_frame = True # RED axis represents x-axis, GREEN axis represents y-axis, and BLUE axis represents z-axis.
option.mesh_show_back_face   = True
option.light_on = False

for cnt in range(80):
    visualize_and_save(vis, colored_point_cloud, cnt= cnt)
    # visualize_and_save(vis, colored_point_cloud, cnt= 50)

vis.destroy_window()

# Option 2
# o3d.visualization.draw_geometries([colored_point_cloud],
#                                   lookat= np.array([0, 0, -10]),
#                                   zoom= 0.8,
#                                   front=  np.array([1, 0, 0]),
#                                   up=     np.array([0, 0, 1.]),
#                                   )
