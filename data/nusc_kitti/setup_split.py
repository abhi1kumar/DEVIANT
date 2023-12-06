import numpy as np
import sys
import os
import shutil
import re

# stop python from writing so much bytecode
sys.dont_write_bytecode = True
sys.path.append(os.getcwd())
np.set_printoptions(suppress=True)

def mkdir_if_missing(directory, delete_if_exist=False):
    """
    Recursively make a directory structure even if missing.

    if delete_if_exist=True then we will delete it first
    which can be useful when better control over initialization is needed.
    """

    if delete_if_exist and os.path.exists(directory): shutil.rmtree(directory)

    # check if not exist, then make
    if not os.path.exists(directory):
        os.makedirs(directory)

def make_symlink_or_copy(src_path, intended_path, MAKE_SYMLINK = True):
    if not os.path.exists(intended_path):
        if MAKE_SYMLINK:
            os.symlink(src_path, intended_path)
        else:
            command = "cp " + src_path + " " + intended_path
            os.system(command)

def link_original_data(org_file_path, new_file_path, org_split_folders, new_split_folders):
    # mkdirs
    mkdir_if_missing(new_split_folders['cal'])
    mkdir_if_missing(new_split_folders['ims'])
    mkdir_if_missing(new_split_folders['lab'])
    # mkdir_if_missing(new_split_folders['dep'])

    print("=> Reading {}".format(org_file_path))
    print("=> Writing {}".format(new_file_path))
    text_file_new = open(new_file_path, 'w')

    text_file     = open(org_file_path, 'r')
    text_lines = text_file.readlines()
    text_file.close()

    for imind, line in enumerate(text_lines):
        parsed = line.strip()

        if parsed is not None:
            id     = parsed
            new_id = '{:06d}'.format(imind)

            org_calib_path = os.path.join(org_split_folders['cal'], id + '.txt')
            org_label_path = os.path.join(org_split_folders['lab'], id + '.txt')
            org_image_path = os.path.join(org_split_folders['ims'], id + '.png')

            # If any of the calib/label/image is missing
            if not os.path.exists(org_calib_path) or not os.path.exists(org_label_path) or not os.path.exists(org_image_path):
                print("{} not found ...".format(parsed))
                imind += 1
                continue

            new_calib_path = os.path.join(new_split_folders['cal'], str(new_id) + '.txt')
            new_label_path = os.path.join(new_split_folders['lab'], str(new_id) + '.txt')
            new_image_path = os.path.join(new_split_folders['ims'], str(new_id) + '.png')

            make_symlink_or_copy(org_calib_path, new_calib_path)
            make_symlink_or_copy(org_label_path, new_label_path)
            make_symlink_or_copy(org_image_path, new_image_path)
            # make_symlink_or_copy(os.path.join(org_split_folders['dep'], id + '.npy'), os.path.join(new_split_folders['dep'], str(new_id) + '.npy'))

            text_file_new.write(new_id + '\n')
            imind += 1

        if imind % 5000 == 0 or line == text_lines[-1]:
            print("{} images done...".format(imind))


    text_file_new.close()

#===================================================================================================
# Main starts here
#===================================================================================================
curr_folder = os.getcwd()
nusc_kitti_org = "nusc_kitti_org"

org_train_folders = dict()
org_train_folders['cal'] = os.path.join(curr_folder, nusc_kitti_org, 'train', 'calib')
org_train_folders['ims'] = os.path.join(curr_folder, nusc_kitti_org, 'train', 'image_2')
org_train_folders['lab'] = os.path.join(curr_folder, nusc_kitti_org, 'train', 'label_2')
# org_train_folders['dep'] = os.path.join(curr_folder, nusc_kitti_org, 'train', 'velodyne')

new_train_folders = dict()
split = "training"
new_train_folders['cal'] = os.path.join(curr_folder, split, 'calib')
new_train_folders['ims'] = os.path.join(curr_folder, split, 'image')
new_train_folders['lab'] = os.path.join(curr_folder, split, 'label')
# new_train_folders['dep'] = os.path.join(curr_folder, split, 'velodyne')

org_val_folders = dict()
org_val_folders['cal'] = os.path.join(curr_folder, nusc_kitti_org, 'val', 'calib')
org_val_folders['ims'] = os.path.join(curr_folder, nusc_kitti_org, 'val', 'image_2')
org_val_folders['lab'] = os.path.join(curr_folder, nusc_kitti_org, 'val', 'label_2')
# org_val_folders['dep'] = os.path.join(curr_folder, nusc_kitti_org, 'val', 'velodyne')


new_val_folders = dict()
split = "validation"
new_val_folders['cal'] = os.path.join(curr_folder, split, 'calib')
new_val_folders['ims'] = os.path.join(curr_folder, split, 'image')
new_val_folders['lab'] = os.path.join(curr_folder, split, 'label')
# new_val_folders['dep'] = os.path.join(curr_folder, split, 'velodyne')

org_train_file = os.path.join(curr_folder, 'ImageSets/train_org.txt')
org_val_file   = os.path.join(curr_folder, 'ImageSets/val_org.txt')
new_train_file = os.path.join(curr_folder, 'ImageSets/train.txt')
new_val_file   = os.path.join(curr_folder, 'ImageSets/val.txt')

#===================================================================================================
# Link train
#===================================================================================================
print('=============== Linking train =======================')
link_original_data(org_file_path= org_train_file, new_file_path= new_train_file, org_split_folders = org_train_folders, new_split_folders= new_train_folders)

#===================================================================================================
# Link val
#===================================================================================================
print('=============== Linking val =======================')
link_original_data(org_file_path= org_val_file, new_file_path= new_val_file, org_split_folders = org_val_folders, new_split_folders= new_val_folders)

print('Done')