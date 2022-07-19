nuscenes_tar_dir="/media/abhinavkumar/baap/abhinavkumar/datasets/nuscenese_tar"  # directory in which tars are there
nusc_output_dir="/media/abhinavkumar/baap/abhinavkumar/datasets/nuscenes"        # directory where we need to untar nuscenes
nusc_kitti_dir="/media/abhinavkumar/baap/abhinavkumar/datasets/nusc_kitti"       # directory where we store nuscenes ~40k images in kitti format

project_dir_relative="project"
project_dir=$HOME/$project_dir_relative

#===============================================================================
# Link paths and extract trainval + test tarballs
#===============================================================================
sudo mkdir -p /data/sets
sudo ln -s $nusc_output_dir /data/sets/nuscenes

cd $nuscenes_tar_dir

# Extract RGB
printf "Extracting RGB images...\n"
tar -xzf v1.0-trainval01_blobs_camera.tgz -C $nusc_output_dir
tar -xzf v1.0-trainval02_blobs_camera.tgz -C $nusc_output_dir
tar -xzf v1.0-trainval03_blobs_camera.tgz -C $nusc_output_dir
tar -xzf v1.0-trainval04_blobs_camera.tgz -C $nusc_output_dir
tar -xzf v1.0-trainval05_blobs_camera.tgz -C $nusc_output_dir
tar -xzf v1.0-trainval06_blobs_camera.tgz -C $nusc_output_dir
tar -xzf v1.0-trainval07_blobs_camera.tgz -C $nusc_output_dir
tar -xzf v1.0-trainval08_blobs_camera.tgz -C $nusc_output_dir
tar -xzf v1.0-trainval09_blobs_camera.tgz -C $nusc_output_dir
tar -xzf v1.0-trainval10_blobs_camera.tgz -C $nusc_output_dir

tar -xzf v1.0-test_blobs_camera.tgz       -C $nusc_output_dir

# Extract Lidar
printf "Extracting Lidar...\n"
tar -xzf v1.0-trainval01_blobs_lidar.tgz -C $nusc_output_dir
tar -xzf v1.0-trainval02_blobs_lidar.tgz -C $nusc_output_dir
tar -xzf v1.0-trainval03_blobs_lidar.tgz -C $nusc_output_dir
tar -xzf v1.0-trainval04_blobs_lidar.tgz -C $nusc_output_dir
tar -xzf v1.0-trainval05_blobs_lidar.tgz -C $nusc_output_dir
tar -xzf v1.0-trainval06_blobs_lidar.tgz -C $nusc_output_dir
tar -xzf v1.0-trainval07_blobs_lidar.tgz -C $nusc_output_dir
tar -xzf v1.0-trainval08_blobs_lidar.tgz -C $nusc_output_dir
tar -xzf v1.0-trainval09_blobs_lidar.tgz -C $nusc_output_dir
tar -xzf v1.0-trainval10_blobs_lidar.tgz -C $nusc_output_dir

tar -xzf v1.0-test_blobs_lidar.tgz       -C $nusc_output_dir

printf "Done\n"

#===============================================================================
# Create conda and set pythonpath
#===============================================================================
conda create --name nuscenes python=3.7
conda activate nuscenes

# Make sure we use full path for $project_dir
export PYTHONPATH="${PYTHONPATH}:$project_dir/nuscenes-devkit/python-sdk"

#===============================================================================
# Download the devkit, move to devkit directory and export train/val/test in 
# kitti format
#===============================================================================
cd $project_dir
git clone https://github.com/abhi1kumar/nuscenes-devkit
cd nuscenes-devkit
pip install -r setup/requirements.txt

cd python-sdk/nuscenes/scripts/
python export_kitti.py nuscenes_gt_to_kitti --nusc_kitti_dir $nusc_kitti_dir --nusc_version v1.0-mini     --split mini_train --image_count 34149
python export_kitti.py nuscenes_gt_to_kitti --nusc_kitti_dir $nusc_kitti_dir --nusc_version v1.0-trainval --split train --image_count 34149
python export_kitti.py nuscenes_gt_to_kitti --nusc_kitti_dir $nusc_kitti_dir --nusc_version v1.0-trainval --split val   --image_count 34149
python export_kitti.py nuscenes_gt_to_kitti --nusc_kitti_dir $nusc_kitti_dir --nusc_version v1.0-test     --split test  --image_count 34149 


#===============================================================================
# Evaluate nuscenes from output files in KITTI format
#===============================================================================
# Make sure we use full path for $project_dir
export PYTHONPATH="${PYTHONPATH}:$HOME/project/nuscenes-devkit/python-sdk"

cd $project_dir
cd /home/abhinavkumar/project/nuscenes-devkit/python-sdk/nuscenes/scripts/
python export_kitti.py kitti_res_to_nuscenes --nusc_kitti_dir ~/project/prevolution/data/nusc_kitti/ --nusc_version v1.0-trainval     --split val --image_count 34149 --output_dir ~/project/prevolution/output/dn121_groomed_ses_105/results/results_test/data

cd ../eval/detection/
python evaluate.py --version v1.0-trainval --eval_set val --plot_examples 0 --render_curves 0 --result_path /home/abhinavkumar/project/prevolution/output/dn121_groomed_ses_105/results/results_test/submission.json --output_dir /home/abhinavkumar/project/prevolution/output/dn121_groomed_ses_105/results/results_test/

#===============================================================================
# Evaluate nuscenes without going to devkit folder
#===============================================================================
# Make sure we use full path for $project_dir
export PYTHONPATH="${PYTHONPATH}:$HOME/project/nuscenes-devkit/python-sdk"

python /home/abhinavkumar/project/nuscenes-devkit/python-sdk/nuscenes/scripts/export_kitti.py kitti_res_to_nuscenes \
--nusc_version v1.0-trainval --split val --image_count 6019 \
--nusc_kitti_dir ~/project/GUPNet/code/data/nusc_kitti/ \
--output_dir     output/run_11/results_140/data

python /home/abhinavkumar/project/nuscenes-devkit/python-sdk/nuscenes/eval/detection/evaluate.py \
--version v1.0-trainval --eval_set val --plot_examples 0 --render_curves 0 \
--result_path ~/project/GUPNet/code/nusc_kitti/submission.json \
--output_dir  ~/project/GUPNet/code/nusc_kitti
