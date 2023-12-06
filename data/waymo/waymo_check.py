import tensorflow as tf
from waymo_open_dataset import dataset_pb2 as open_dataset

tfrecord_file_name = '/media/abhinav/baap/abhinav/datasets/waymo_open_tfrecord/validation/segment-967082162553397800_5102_900_5122_900_with_camera_labels.tfrecord'  # give one tfrecord here

def read_tfrecord():
    dataset = tf.data.TFRecordDataset(tfrecord_file_name, compression_type='')
    for data in dataset:
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        for obj in frame.laser_labels:
             print(obj.num_lidar_points_in_box)

if __name__ == '__main__':
    read_tfrecord()
