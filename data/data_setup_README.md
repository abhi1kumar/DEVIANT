## Data Setup

Download the full [KITTI 3D Object detection](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) dataset:
    
- [left color images](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip) of object data set (12 GB)
- [camera calibration matrices](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_calib.zip)  of object data set (16 MB)
- [training labels](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip) of object data set (5 MB)

Download the [nuScenes](https://www.nuscenes.org/nuscenes#download) and [Waymo](https://waymo.com/open/download/) datasets. Note- Both these datasets are quite big and requires about 1TB storage space each.

Setup KITTI, nuScenes and Waymo as follows:

```bash
./code
├── data
│      ├── KITTI
│      │      ├── ImageSets
│      │      ├── kitti_split1
│      │      ├── training
│      │      │     ├── calib
│      │      │     ├── image_2
│      │      │     └── label_2
│      │      │
│      │      └── testing
│      │            ├── calib
│      │            └── image_2
│      │
│      ├── nusc_kitti
│      │      ├── ImageSets
│      │      └── nuscenes
│      │            ├── maps
│      │            ├── samples
│      │            └── v1.0-trainval
│      │
│      └── waymo
│             ├── ImageSets
│             └── raw_data
│                   ├── training
│                   └── validation
│
├── experiments
├── images
├── lib
├── nuscenes-devkit        
│ ...
```

### KITTI

Simply put the soft links:

```bash
cd data/KITTI/
ln -sfn your_path/kitti/training training
ln -sfn your_path/kitti/testing testing
cd ../..
```

### nuScenes

Next download the patched nuScenes devkit. This patched devkit provides lidar points per box in KITTI format and also supports evaluation on nuScenes front camera:

```bash
git clone https://github.com/abhi1kumar/nuscenes-devkit
```

Then follow the instructions at [convert_nuscenes_to_kitti_format_and_evaluate.sh](nusc_kitti/convert_nuscenes_to_kitti_format_and_evaluate.sh) to get `nusc_kitti_org` folder. Finally link this folder with the following command: 

```bash
cd data/nusc_kitti/
ln -sfn your_path/nusc_kitti_org nusc_kitti_org
```

This generates the following structure:

```bash
./code
├── data
│      ├── nusc_kitti
│      │      ├── ImageSets
│      │      ├── nusc_kitti_org
│      │      │     ├── train
│      │      │     │     ├── calib
│      │      │     │     ├── image_2
│      │      │     │     └── label_2
│      │      │     │    
│      │      │     └── val
│      │      │           ├── calib
│      │      │           ├── image_2
│      │      │           └── label_2
│      │      └── nuscenes
│      │            ├── maps
│      │            ├── samples
│      │            └── v1.0-trainval
│      │       
```

Finally run the script `setup_split.py` to generate `training` and `validation` folders of the nuScenes Val split:

```bash
ln -sfn your_path/nusc_kitti_training_mapped training
ln -sfn your_path/nusc_kitti_validation_mapped validation
python setup_split.py
cd ../..
```

The script uses soft-links for efficient storage and creates the desired directory structure for nuScenes Val, which can be used by any KITTI based detector.

```bash
./code
├── data
│      ├── nusc_kitti
│      │      ├── ImageSets
│      │      ├── nusc_kitti_org
│      │      │     ├── train
│      │      │     │     ├── calib
│      │      │     │     ├── image_2
│      │      │     │     └── label_2
│      │      │     │    
│      │      │     └── val
│      │      │           ├── calib
│      │      │           ├── image_2
│      │      │           └── label_2
│      │      ├── nuscenes
│      │      │     ├── maps
│      │      │     ├── samples
│      │      │     └── v1.0-trainval
│      │      │
│      │      ├── training
│      │      │     ├── calib
│      │      │     ├── image
│      │      │     └── label
│      │      │
│      │      └── validation
│      │            ├── calib
│      │            ├── image
│      │            └── label
```

### Waymo

Decompress the Waymo zip files into their corresponding directories:

```bash
ls *.tar | xargs -i tar xvf {} -C your_target_dir
```

Each directory contains tfrecords. Arrange them in the following fashion:

```bash
./code
├── data
│      └── waymo
│             ├── ImageSets
│             └── raw_data
│                   ├── training
│                   │     ├── segment-10017090168044687777_6380_000_6400_000_with_camera_labels.tfrecord
│                   │     └── segment-10023947602400723454_1120_000_1140_000_with_camera_labels.tfrecord
│                   │
│                   └── validation
│                         ├── segment-10203656353524179475_7625_000_7645_000_with_camera_labels.tfrecord
│                         └── segment-1024360143612057520_3580_000_3600_000_with_camera_labels.tfrecord
```

Then, setup the Waymo devkit. The Waymo devkit is setup in a different environment to avoid package conflicts with our DEVIANT environment:

```bash
# Set up environment
conda create -n py36_waymo_tf python=3.7
conda activate py36_waymo_tf
conda install cudatoolkit=11.3 -c pytorch

# Newer versions of tf are not in conda. tf>=2.4.0 is compatible with conda.
pip install tensorflow-gpu==2.4
conda install pandas
pip3 install waymo-open-dataset-tf-2-4-0 --user
```

Next convert the segments to the KITTI format using `converter.py`. Note that we have commented out the code for saving lidar frames which takes huge amount of time. Make sure to keep the number of processes `num_proc` to the highest number supported by your GPU. The conversion takes about 3 days to complete on our end.

```bash
conda activate py36_waymo_tf
cd data/waymo/
python converter.py --load_dir "" --save_dir your_path/datasets/waymo_open_organized/ --split training   --num_proc 10
python converter.py --load_dir "" --save_dir your_path/datasets/waymo_open_organized/ --split validation --num_proc 10
ln -sfn your_path/datasets/waymo_open_organized/training_org training_org
ln -sfn your_path/datasets/waymo_open_organized/validation_org validation_org
```

This will result in the following directory structure:

```bash
./code
├── data
│      └── waymo
│             ├── ImageSets
│             ├── raw_data
│             │     ├── training
│             │     │     ├── segment-10017090168044687777_6380_000_6400_000_with_camera_labels.tfrecord
│             │     │     └── segment-10023947602400723454_1120_000_1140_000_with_camera_labels.tfrecord
│             │     │
│             │     └── validation
│             │           ├── segment-10203656353524179475_7625_000_7645_000_with_camera_labels.tfrecord
│             │           └── segment-1024360143612057520_3580_000_3600_000_with_camera_labels.tfrecord
│             │
│             ├── training_org
│             │     ├── segment_id
│             │           ├── calib
│             │           ├── image_0
│             │           ├── image_1
│             │           ├── image_2
│             │           ├── image_3
│             │           ├── image_4
│             │           ├── label_0
│             │           ├── label_1
│             │           ├── label_2
│             │           ├── label_3
│             │           ├── label_4
│             │           ├── label_all
│             │           ├── projected_points_0
│             │           └── velodyne
│             │
│             └── validation_org
│                   ├── segment_id
│                         ├── calib
│                         ├── image_0
│                         ├── image_1
│                         ├── image_2
│                         ├── image_3
│                         ├── image_4
│                         ├── label_0
│                         ├── label_1
│                         ├── label_2
│                         ├── label_3
│                         ├── label_4
│                         ├── label_all
│                         ├── projected_points_0
│                         └── velodyne

```

As a sanity check, you should see a total of more than 39k  `calib`, `image_0` and `label_0` files in all the segment_ids of `validation_org` folder.

```bash
cd validation_org
find */calib -type f | wc -l
find */image_0 -type f | wc -l
find */label_0 -type f | wc -l
cd ..
```

Finally, run the script `setup_split.py` to convert the segments into standard KITTI format and generate `training` and `validation` folders of the Waymo Val split: 

```bash
python setup_split.py
```

The script uses soft-links for efficient storage and creates the desired directory structure, which can be used by any KITTI based detector.

```bash
./code
├── data
│      └── waymo
│             ├── ImageSets
│             ├── raw_data
│             │     ├── training
│             │     └── validation
│             │
│             ├── training_org
│             ├── validation_org
│             │
│             ├── training
│             │     ├── calib
│             │     ├── image
│             │     └── label
│             │
│             └── validation
│                   ├── calib
│                   ├── image
│                   └── label

```

Also prepare a small val split for testing:

```bash
cd ImageSets
sort -R --random-source=<(yes 123) val.txt | head -n 1000 > val_small.txt
cd ../../../
```

**Alter Way**

We also upload the `calib` and `label` subfolders of the waymo training and validation split at this [ drive link](https://drive.google.com/file/d/1Yzs3gZWsdvI0IX_ZjeiI5Y0XmrFutfFz/view?usp=sharing).

Unzip the above file and place them as follows:
```bash
DEVIANT
├── data
│      └── waymo
│             ├── ImageSets
│             ├── training
│             │     ├── calib
│             │     └── label
│             │
│             └── validation
│                   ├── calib
│                   └── label

```

Then consider copying the corresponding images in the `image` sub-folder of `training` and `validation` folders to complete the folder structure:
```bash
DEVIANT
├── data
│      └── waymo
│             ├── ImageSets
│             ├── training
│             │     ├── calib
│             │     ├── image
│             │     └── label
│             │
│             └── validation
│                   ├── calib
│                   ├── image
│                   └── label

```



**Transfer**

If you have Waymo dataset prepared and you need to transfer to your server, type the following:

```bash
cd your_desktop_waymo_location
cd training_org
rsync -qavR */calib/ abhinavkumar@rsync.hpcc.msu.edu:/mnt/gs21/scratch/abhinavkumar/data/waymo_open_organized/training_org
rsync -qavR */image_0/ abhinavkumar@rsync.hpcc.msu.edu:/mnt/gs21/scratch/abhinavkumar/data/waymo_open_organized/training_org
rsync -qavR */label_0/ abhinavkumar@rsync.hpcc.msu.edu:/mnt/gs21/scratch/abhinavkumar/data/waymo_open_organized/training_org

cd ../validation_org
rsync -qavR */calib/ abhinavkumar@rsync.hpcc.msu.edu:/mnt/gs21/scratch/abhinavkumar/data/waymo_open_organized/validation_org
rsync -qavR */image_0/ abhinavkumar@rsync.hpcc.msu.edu:/mnt/gs21/scratch/abhinavkumar/data/waymo_open_organized/validation_org
rsync -qavR */label_0/ abhinavkumar@rsync.hpcc.msu.edu:/mnt/gs21/scratch/abhinavkumar/data/waymo_open_organized/validation_org
```
