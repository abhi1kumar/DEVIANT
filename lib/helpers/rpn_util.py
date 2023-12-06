import os,sys
import re
from easydict import EasyDict as edict
import subprocess
import glob
import torch

from lib.helpers.file_io import *
from lib.helpers.util import *

def evaluate_nusc_kitti_results_verbose(results_folder, conf= None, use_logging= False, logger= None):
    # Make symlinks id.txt --> numbers.txt
    predictions          = sorted(glob.glob(results_folder + "/*.txt"))
    val_org_samples      = read_lines(path = "data/nusc_kitti/ImageSets/val_org.txt")
    results_folder_new   = results_folder.replace("data", "data2")
    mkdir_if_missing(results_folder_new, delete_if_exist= True)
    for i in range(len(val_org_samples)):
        src_path = predictions[i]
        intended_path = os.path.join(results_folder_new, val_org_samples[i] + ".txt")
        os.symlink( os.path.join(os.getcwd(), src_path), os.path.join(os.getcwd(), intended_path))

    # Find home in path
    # Typically the conda environment is /home/abc/anaconda3/envs/nuscenes or /mnt/home/abc/anaconda3/envs/nuscenes
    curr_folder      = os.getcwd().split("/")
    home_word_index  = [idx for idx, s in enumerate(curr_folder) if 'home' in s][0]
    home_user_folder = "/".join(curr_folder[:(home_word_index+2)])
    python_path      = os.path.join(home_user_folder, "anaconda3/envs/nuscenes/bin/python")  #/home/abc/anaconda3/envs/nuscenes/

    # path of the nuscenes-devkit
    devkit_path      = os.path.join(os.getcwd(), "nuscenes-devkit/python-sdk/")              #/home/abc/project/DEVIANT/code/nuscenes-devkit/python-sdk
    devkit_path_more = os.path.join(devkit_path, "nuscenes")                                 #/home/abc/project/DEVIANT/code/nuscenes-devkit/python-sdk/nuscenes

    # Convert to JSON format first
    output_folder    = os.path.dirname(results_folder_new)
    json_folder      = output_folder
    json_path        = os.path.join(json_folder, "submission.json")
    command_convert  = python_path + " " + os.path.join(devkit_path_more, "scripts/export_kitti.py")\
                       + " kitti_res_to_nuscenes --nusc_version v1.0-trainval --split val --image_count 6019 "\
                       + " --nusc_kitti_dir " + "data/nusc_kitti/nusc_kitti_org"\
                       + " --output_dir " + results_folder_new

    # Run nuScenes Evaluation
    command_evaluate = python_path + " " + os.path.join(devkit_path_more, "eval/detection/evaluate.py")\
                       + " --version v1.0-trainval --eval_set val --plot_examples 0 --render_curves 0"\
                       + " --result_path " + json_path\
                       + " --output_dir " + json_folder\
                       + " --verbose 0 --use_only_cam_front_gt 1"

    # export PYTHONPATH is also needed.
    # If we use sys.append, it does not get passed on to subprocess.check_output
    command_export   = "export PYTHONPATH=\"${PYTHONPATH}:" + devkit_path + "\";"

    if use_logging:
        logger.info("Converting to JSON...")
        logger.info(command_export + command_convert)
        try:
            logger.info(safe_subprocess(command_export + command_convert))
            safe_subprocess("mv " + os.path.join(results_folder_new, "submission.json") + " " + output_folder)
            logger.info("Running nuScenes evaluation...")
            logger.info(command_export + command_evaluate)
            logger.info(safe_subprocess(command_export + command_evaluate))
            safe_subprocess("rm -rf " + results_folder_new)
        except:
            logger.info("An exception occurred while nusc_kitti evaluation ...")
    else:
        print("Converting to JSON...")
        print(command_export + command_convert)
        try:
            print(safe_subprocess(command_export + command_convert))
            print("Running nuScenes evaluation...")
            print(command_export + command_evaluate)
            print(safe_subprocess(command_export + command_evaluate))
            safe_subprocess("rm -rf " + results_folder_new)
        except:
            print("An exception occurred while nusc_kitti evaluation ...")

def get_MAE(results_folder, gt_folder, conf= None, use_logging= False, logger= None, \
                                      thresholds= np.array([0, 20, 40, 1000]), \
                                      iou_overlap_threshold= 0.7):
    num_thresholds = thresholds.shape[0]
    error_box    = []
    for i in range(num_thresholds):
        error_box.append([])
    pred_box_cnt      = np.zeros(thresholds.shape[0])
    gt_box_cnt         = np.zeros(thresholds.shape[0])

    # Read all pred files
    pred_files    = sorted(glob.glob(results_folder + "/*.txt"))
    num_images    = len(pred_files)
    if use_logging:
        logger.info("Found {} predictions in {}, GT= {}".format(num_images, results_folder, gt_folder))
    else:
        print("Found {} predictions in {}, GT= {}".format(num_images, results_folder, gt_folder))

    for i in range(num_images):
        pred_file       = pred_files[i]

        basename_wo_ext = os.path.basename(pred_files[i]).split(".")[0]
        gt_file         = os.path.join(gt_folder   , basename_wo_ext + ".txt")

        # Read ground truth and filter by cars
        gt_boxes_orig   = read_csv(path= gt_file  , delimiter= " ", ignore_warnings= False, use_pandas= True)
        if gt_boxes_orig is None:
            continue
        gt_boxes        = filter_boxes_by_cars(gt_boxes_orig)
        if gt_boxes.shape[0] == 0:
            continue

        # GT boxes data
        for j in range(1, thresholds.shape[0]):
            valid_gt_index_for_th  = np.logical_and(gt_boxes >= thresholds[j-1], gt_boxes < thresholds[j])
            num_valid_gt_boxes_th  = np.sum(valid_gt_index_for_th)
            gt_box_cnt[j]         += num_valid_gt_boxes_th
            gt_box_cnt[0]         += num_valid_gt_boxes_th

        # Read predictions and filter by cars
        pred_boxes_orig = read_csv(path= pred_file, delimiter= " ", ignore_warnings= False, use_pandas= True)
        if pred_boxes_orig is None:
            continue
        pred_boxes      = filter_boxes_by_cars(pred_boxes_orig)
        if pred_boxes.shape[0] == 0:
            continue

        # Now get an IoU between them
        pred_x1_y1_x2_y2, pred_z = get_x1_y1_x2_y2_z(pred_boxes)
        gt_x1_y1_x2_y2  , gt_z   = get_x1_y1_x2_y2_z(gt_boxes)
        iou_mat                  = iou(pred_x1_y1_x2_y2, gt_x1_y1_x2_y2)

        # Get max for each box
        max_gt_overlap = np.max   (iou_mat, axis= 1)
        max_gt_index   = np.argmax(iou_mat, axis= 1)

        # Check if it there is sufficient overlap
        valid_index    = max_gt_overlap >= iou_overlap_threshold
        num_valid_boxes= np.sum(valid_index)

        if num_valid_boxes > 0:
            pred_boxes_valid    = pred_boxes[valid_index]
            pred_z_valid        = pred_z[valid_index]
            pred_boxes_valid_gt = gt_boxes[max_gt_index][valid_index]
            gt_z_valid          = gt_z[max_gt_index][valid_index]

            error_boxes_valid     = np.abs(pred_z_valid - gt_z_valid)

            # All boxes data
            error_box[0] = custom_append(storage= error_box[0], curr= error_boxes_valid)
            pred_box_cnt[0]   += num_valid_boxes

            # GT binned threshold boxes data
            for j in range(1, thresholds.shape[0]):
                valid_index_for_th  = np.logical_and(gt_z_valid >= thresholds[j-1], gt_z_valid < thresholds[j])
                num_valid_boxes_th  = np.sum(valid_index_for_th)

                if num_valid_boxes_th > 0:
                    error_box[j] = custom_append(storage= error_box[j], curr= error_boxes_valid[valid_index_for_th])
                    pred_box_cnt[j]   += num_valid_boxes_th

    error_avg = np.zeros((num_thresholds, ))
    error_med = np.zeros((num_thresholds, ))
    for i in range(num_thresholds):
        if pred_box_cnt[i] > 0:
            error_avg[i] = np.sum(error_box[i]) / (pred_box_cnt[i] + 1e-6)
            error_med[i] = np.median(error_box[i])

    # Cicular permute the matrix
    error_avg    = np.roll(error_avg,-1)
    error_med    = np.roll(error_med,-1)
    pred_box_cnt = np.roll(pred_box_cnt, -1).astype(int)
    gt_box_cnt   = np.roll(gt_box_cnt   ,-1).astype(int)

    # Prepare log_str
    log_str  = 'Running MAE Statistics...\n'
    log_str += '----------------------------------------------------------------------------------\n'
    log_str += '       Avg              |         Median         |      # boxes / #total boxes\n'
    for j in range(3):
        for i in range(num_thresholds):
            if i == num_thresholds-2:
                log_str += "{:2d}+   ".format(thresholds[i])
            elif i == num_thresholds-1:
                log_str += "All   |"
            else:
                log_str += "{:2d}-{:2d} ".format(thresholds[i], thresholds[i+1])
    log_str += "\n"
    log_str += '----------------------------------------------------------------------------------\n'
    for i in range(num_thresholds):
        log_str += '{:.3f} '.format(error_avg[i])
    log_str += "| "
    for i in range(num_thresholds):
        log_str += '{:.3f} '.format(error_med[i])
    log_str += "| "
    for i in range(num_thresholds):
        log_str += '{:d}/{:d} '.format(pred_box_cnt[i], gt_box_cnt[i])
    log_str += "\n"
    # for i in range(num_thresholds):
    #     log_str += '{:06d} '.format()

    if use_logging:
        logger.info(log_str)
    else:
        print(log_str)

def evaluate_waymo_results_verbose(results_folder, conf= None, use_logging= False, logger= None):
    # Find home in path
    # Typically the conda environment is /home/abc/anaconda3/envs/py36_waymo_tf/ or /mnt/home/abc/anaconda3/envs/py36_waymo_tf/
    curr_folder      = os.getcwd().split("/")
    home_word_index  = [idx for idx, s in enumerate(curr_folder) if 'home' in s][0]
    home_user_folder = "/".join(curr_folder[:(home_word_index+2)])

    path             = os.path.join(home_user_folder, "anaconda3/envs/py36_waymo_tf/bin/python") #/home/abc/anaconda3/envs/py36_waymo_tf/
    pd_set           = os.path.join("data/waymo/ImageSets", conf['dataset']['val_split_name']+ ".txt")

    command_0_7      = path + " -u data/waymo/waymo_eval.py --predictions "     + results_folder + " --pd_set " + pd_set
    command_0_5      = path + " -u data/waymo/waymo_eval_0_5.py --predictions " + results_folder + " --pd_set " + pd_set

    if use_logging:
        logger.info("Running for thresholds [0.7, 0.5, 0.5]...")
        logger.info(subprocess.check_output(command_0_7, shell= True, text= True))
        logger.info("Running for thresholds [0.5, 0.3, 0.3]...")
        logger.info(subprocess.check_output(command_0_5, shell= True, text= True))
    else:
        print("Running for thresholds [0.7, 0.5, 0.5]...")
        print(subprocess.check_output(command_0_7, shell= True, text= True))
        print("Running for thresholds [0.5, 0.3, 0.3]...")
        print(subprocess.check_output(command_0_5, shell= True, text= True))

def evaluate_kitti_results_verbose(data_folder, test_dataset_name, results_folder, split_name= "validation", test_iter= None, conf= None, use_logging=False, logger= None, fast=True, default_eval='evaluate_object'):
    """

    :param data_folder: folder containing ground truth
    :param test_dataset_name: kitti_split1, kitti_split2
    :param results_folder: folder which has the results. Usually it is output/config/results/results_test/data
    :param split_name: split to use- validation/training
    :param test_iter: 70000
    :param conf: conf easydict
    :param use_logging:
    :param fast:
    :return:
    """

    stats_save_folder = os.path.dirname(results_folder)
    gt_folder         = "data/KITTI/training/label_2" #os.path.join(data_folder, test_dataset_name, split_name, "label_2")
    devkit_folder     = "data/KITTI/"#.replace("data", "splits_prep")

    # evaluate primary experiment
    binary_test_dataset_name = "kitti_split1" # test_dataset_name   # Use binaries of kitti_split1
    eval_binary_path = os.path.join(devkit_folder, binary_test_dataset_name, 'devkit', 'cpp', default_eval)

    results_obj = edict()

    task_keys = ['det_2d', 'or', 'gr', 'det_3d']
    lbls = conf['dataset']['writelist']

    # main experiment results
    results_obj.main = run_kitti_eval_script(eval_binary_path, results_data= stats_save_folder, gt_folder= gt_folder, lbls= lbls, use_40=True)
    print("")
    if use_logging:
        logger.info("Running for thresholds [0.7, 0.5, 0.5]...")
    # print main experimental results for each class, and each task
    for lbl in lbls:

        lbl = lbl.lower()

        for task in task_keys:

            task_lbl = task + '_' + lbl

            if task_lbl in results_obj.main:

                easy = results_obj.main[task_lbl][0]
                mod = results_obj.main[task_lbl][1]
                hard = results_obj.main[task_lbl][2]

                task_print = task.replace('det_', '')
                if task_print == 'gr':
                    task_print = 'bev'
                print_str = 'test_iter {} {} {:3s} --> easy: {:0.4f}, mod: {:0.4f}, hard: {:0.4f}'\
                    .format(test_iter, lbl, task_print, easy, mod, hard)

                if use_logging:
                    logger.info(print_str)
                else:
                    print(print_str)

    # side experiment results
    eval_binary_path = os.path.join(devkit_folder, binary_test_dataset_name, 'devkit', 'cpp', default_eval + "_0_5")
    results_obj.side = run_kitti_eval_script(eval_binary_path, results_data= stats_save_folder, gt_folder= gt_folder, lbls= lbls, use_40=True)
    print("")
    if use_logging:
        logger.info("Running for thresholds [0.5, 0.3, 0.3]...")
    # print main experimental results for each class, and each task
    for lbl in lbls:

        lbl = lbl.lower()

        for task in task_keys:

            task_lbl = task + '_' + lbl

            if task_lbl in results_obj.side:

                easy = results_obj.side[task_lbl][0]
                mod = results_obj.side[task_lbl][1]
                hard = results_obj.side[task_lbl][2]

                task_print = task.replace('det_', '')
                if task_print == 'gr':
                    task_print = 'bev'
                print_str = 'test_iter {} {} {:3s} --> easy: {:0.4f}, mod: {:0.4f}, hard: {:0.4f}'\
                    .format(test_iter, lbl, task_print, easy, mod, hard)

                if use_logging:
                    logger.info(print_str)
                else:
                    print(print_str)

    if fast: return

    iou_keys = ['0_1', '0_2', '0_3', '0_4', '0_5', '0_6', '0_7']
    dis_keys = ['15' , '30', '45', '60']

    for k, dis_key in enumerate(dis_keys):
        if k== 0:
            print("Getting AP3D at        ground truth distance <= {}m".format(dis_key), end= '', flush= True)
        else:
            print("Getting AP3D at {}m <= ground truth distance <= {}m".format(dis_keys[k-1], dis_key), end='', flush=True)

        for iou_key in iou_keys:
            eval_key = 'evaluate_object_{}m_{}'.format(dis_key, iou_key)
            save_key = 'res_{}m_{}'.format(dis_key, iou_key)

            print('.', end= '', flush= True)
            eval_binary_path      = os.path.join(devkit_folder, binary_test_dataset_name, 'devkit', 'cpp', eval_key)
            tmp_obj               = run_kitti_eval_script(eval_binary_path, results_data= stats_save_folder, gt_folder= gt_folder, lbls= lbls, use_40= True)
            results_obj[save_key] = tmp_obj

        print('', flush= True)

    pickle_write(os.path.join(stats_save_folder, 'AP_vs_IOU3D_threshold_at_different_gt_distances.pkl'), results_obj)


def run_kitti_eval_script(eval_binary_path, results_data, gt_folder, lbls, use_40=True):

    # evaluate primary experiment
    with open(os.devnull, 'w') as devnull:
        _ = subprocess.check_output([eval_binary_path, results_data, gt_folder], stderr=devnull)

    results_obj = edict()

    for lbl in lbls:

        lbl = lbl.lower()

        respath_2d = os.path.join(results_data, 'stats_{}_detection.txt'.format(lbl))
        respath_or = os.path.join(results_data, 'stats_{}_orientation.txt'.format(lbl))
        respath_gr = os.path.join(results_data, 'stats_{}_detection_ground.txt'.format(lbl))
        respath_3d = os.path.join(results_data, 'stats_{}_detection_3d.txt'.format(lbl))

        if os.path.exists(respath_2d):
            easy, mod, hard = parse_kitti_result(respath_2d, use_40=use_40)
            results_obj['det_2d_' + lbl] = [easy, mod, hard]
        if os.path.exists(respath_or):
            easy, mod, hard = parse_kitti_result(respath_or, use_40=use_40)
            results_obj['or_' + lbl] = [easy, mod, hard]
        if os.path.exists(respath_gr):
            easy, mod, hard = parse_kitti_result(respath_gr, use_40=use_40)
            results_obj['gr_' + lbl] = [easy, mod, hard]
        if os.path.exists(respath_3d):
            easy, mod, hard = parse_kitti_result(respath_3d, use_40=use_40)
            results_obj['det_3d_' + lbl] = [easy, mod, hard]

    return results_obj

def parse_kitti_result(respath, use_40=False):
    acc = np.zeros([3, 41], dtype=float)
    text_file = open(respath, 'r')
    lind = 0
    for line in text_file:
        parsed = re.findall('([\d]+\.?[\d]*)', line)
        for i, num in enumerate(parsed):
            acc[lind, i] = float(num)
        lind += 1
    text_file.close()

    if use_40:
        easy = np.mean(acc[0, 1:41:1])
        mod = np.mean(acc[1, 1:41:1])
        hard = np.mean(acc[2, 1:41:1])
    else:
        easy = np.mean(acc[0, 0:41:4])
        mod = np.mean(acc[1, 0:41:4])
        hard = np.mean(acc[2, 0:41:4])

    return easy, mod, hard

def combine(storage, new_entry):
    if storage is None or len(storage) == 0:
        storage = new_entry
    elif type(storage) == np.ndarray:
        if storage.ndim > 1:
            storage = np.vstack((storage, new_entry))
        else:
            storage = np.hstack((storage, new_entry))
    return storage

def custom_append(storage, curr):
    if type(storage) == np.ndarray:
        storage = np.append(storage, curr)
    else:
        storage = curr
    return storage

def safe_subprocess(command):
    # return subprocess.check_output(command, shell= True, text= True)
    try:
        # https://stackoverflow.com/a/22582602
        # Alter: start_new_session=True https://alexandra-zaharia.github.io/posts/kill-subprocess-and-its-children-on-timeout-python/
        proc = subprocess.Popen([command], preexec_fn=os.setsid, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell= True)
        output, error = proc.communicate()
        return output.decode('UTF-8')
    except:
        return "Error Occurred"

def get_x1_y1_x2_y2_z(boxes):
    #       0  1   2      3   4   5   6    7    8    9   10   11   12
    # cls, -1, -1, alpha, x1, y1, x2, y2, h3d, w3d, l3d, x3d, y3d, z3d, ry3d, score
    return boxes[:, 3:7], boxes[:, 12]

def filter_boxes_by_cars(boxes):
    mask = np.logical_or(boxes[:, 0]== "Car", boxes[:, 0]== "car")
    return boxes[mask][:, 1:]

def intersect(box_a, box_b, mode='combinations', data_type=None):
    """
    Computes the amount of intersect between two different sets of boxes.

    Args:
        box_a (nparray): Mx4 boxes, defined by [x1, y1, x2, y2]
        box_a (nparray): Nx4 boxes, defined by [x1, y1, x2, y2]
        mode (str): either 'combinations' or 'list', where combinations will check all combinations of box_a and
                    box_b hence MxN array, and list expects the same size list M == N, hence returns Mx1 array.
        data_type (type): either torch.Tensor or np.ndarray, we automatically determine otherwise
    """

    # determine type
    if data_type is None: data_type = type(box_a)

    # this mode computes the intersect in the sense of combinations.
    # i.e., box_a = M x 4, box_b = N x 4 then the output is M x N
    if mode == 'combinations':

        # np.ndarray
        if data_type == np.ndarray:
            # print("Using combinations in intersect function on numpy")
            # Calculates the coordinates of the overlap box
            # eg if the box x-coords is at 4 and 5, then the overlap will be minimum
            # of the two which is 4
            # np.maximum is to take two arrays and compute their element-wise maximum.
            # Here, 'compatible' means that one array can be broadcast to the other.
            max_xy = np.minimum(box_a[:, 2:4], np.expand_dims(box_b[:, 2:4], axis=1))
            min_xy = np.maximum(box_a[:, 0:2], np.expand_dims(box_b[:, 0:2], axis=1))
            inter = np.clip((max_xy - min_xy), a_min=0, a_max=None)

        elif data_type == torch.Tensor:
            max_xy = torch.min(box_a[:, 2:4], box_b[:, 2:4].unsqueeze(1))
            min_xy = torch.max(box_a[:, 0:2], box_b[:, 0:2].unsqueeze(1))
            inter = torch.clamp((max_xy - min_xy), 0)

        # unknown type
        else:
            raise ValueError('type {} is not implemented'.format(data_type))

        return inter[:, :, 0] * inter[:, :, 1]

    # this mode computes the intersect in the sense of list_a vs. list_b.
    # i.e., box_a = M x 4, box_b = M x 4 then the output is Mx1
    elif mode == 'list':

        # torch.Tesnor
        if data_type == torch.Tensor:
            max_xy = torch.min(box_a[:, 2:], box_b[:, 2:])
            min_xy = torch.max(box_a[:, :2], box_b[:, :2])
            inter = torch.clamp((max_xy - min_xy), 0)

        # np.ndarray
        elif data_type == np.ndarray:
            max_xy = np.minimum(box_a[:, 2:], box_b[:, 2:])
            min_xy = np.maximum(box_a[:, :2], box_b[:, :2])
            inter = np.clip((max_xy - min_xy), a_min=0, a_max=None)

        # unknown type
        else:
            raise ValueError('unknown data type {}'.format(data_type))

        return inter[:, 0] * inter[:, 1]

    else:
        raise ValueError('unknown mode {}'.format(mode))

def iou(box_a, box_b, mode='combinations', data_type=None):
    """
    Computes the amount of Intersection over Union (IoU) between two different sets of boxes.

    Args:
        box_a (nparray): Mx4 boxes, defined by [x1, y1, x2, y2]
        box_a (nparray): Nx4 boxes, defined by [x1, y1, x2, y2]
        mode (str): either 'combinations' or 'list', where combinations will check all combinations of box_a and
                    box_b hence MxN array, and list expects the same size list M == N, hence returns Mx1 array.
        data_type (type): either torch.Tensor or np.ndarray, we automatically determine otherwise
    """

    # determine type
    if data_type is None: data_type = type(box_a)

    # this mode computes the IoU in the sense of combinations.
    # i.e., box_a = M x 4, box_b = N x 4 then the output is M x N
    if mode == 'combinations':

        inter = intersect(box_a, box_b, data_type=data_type)
        area_a = ((box_a[:, 2] - box_a[:, 0]) *
                  (box_a[:, 3] - box_a[:, 1]))
        area_b = ((box_b[:, 2] - box_b[:, 0]) *
                  (box_b[:, 3] - box_b[:, 1]))

        # torch.Tensor
        if data_type == torch.Tensor:
            union = area_a.unsqueeze(0) + area_b.unsqueeze(1) - inter
            return (inter / union).permute(1, 0)

        # np.ndarray
        elif data_type == np.ndarray:
            union = np.expand_dims(area_a, 0) + np.expand_dims(area_b, 1) - inter
            return (inter / union).T

        # unknown type
        else:
            raise ValueError('unknown data type {}'.format(data_type))


    # this mode compares every box in box_a with target in box_b
    # i.e., box_a = M x 4 and box_b = M x 4 then output is M x 1
    elif mode == 'list':

        inter = intersect(box_a, box_b, mode=mode)
        area_a = (box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])
        area_b = (box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])
        union = area_a + area_b - inter

        return inter / union

    else:
        raise ValueError('unknown mode {}'.format(mode))
