import os
import tensorflow as tf
import numpy as np
from collections import defaultdict
import time
import argparse


class tfrecords_maker:
    def __init__(self, dataset_dir, dataset_name):
        """
        Introduction
        ------------
            构造函数
        Parameters
        ----------
            data_dir: 文件路径
            mode: 数据集模式 "train"
            anchors: 数据集聚类得到的anchor
            num_classes: 数据集图片类别数量
            input_shape: 图像输入模型的大小
            max_boxes: 每张图片最大的box数量
            jitter: 随机长宽比系数
            hue: 调整hsv颜色空间系数
            sat: 调整饱和度系数
            cont: 调整对比度系数
            bri: 调整亮度系数
        """

        self.dataset_name = dataset_name
        self.dataset_dir = dataset_dir

    def convert_to_tfrecord(self, tfrecord_path, num_person, SenceName):
        if not os.path.exists(tfrecord_path):
            os.makedirs(tfrecord_path)
        for idx in range(1, num_person+1):
            datas = np.loadtxt(os.path.join(self.dataset_dir, self.dataset_name, str(idx) + '.txt'), dtype=np.float32)

            output_file = os.path.join(tfrecord_path, str(idx) + '_' + SenceName + '.tfrecords')
            frames = datas.shape[0]
            with tf.python_io.TFRecordWriter(output_file) as record_writer:
                for index in range(frames):
                    example = tf.train.Example(features = tf.train.Features(
                        feature = {'person_data' : tf.train.Feature(float_list = tf.train.FloatList(value = datas[index].tolist()))
                        }))
                    record_writer.write(example.SerializeToString())

                    print('Processed {} experience....'.format(index))
                print(output_file + ' is ok....')

if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument('--SenceName', default='Counterflow_Comp',
                       help='near state dataset save dir')
    parse.add_argument('--DirectionDatasetDir',default='F:/Pedestrians_Data/Non_Reward/Convection',
                       help='near state dataset save dir')
    parse.add_argument('--DirectionDatasetName', default='Convection1/', help='near state dataset name')
    parse.add_argument('--VelocityDatasetDir', default='F:/Pedestrians_Data/Non_Reward/Convection',
                       help='near state dataset save dir')
    parse.add_argument('--VelocityDatasetName', default='Convection1/', help='near state dataset name')
    parse.add_argument('--SaveDirecitonDir', default='F:/Pedestrians_Data_Processed/Non_Reward/Convection',
                       help='experience data save dir')
    parse.add_argument('--SaveDirecitonName', default='Convection1/', help='experience data set name')

    parse.add_argument('--SaveVelocityDir', default='F:/Pedestrians_Data_Processed/Non_Reward/Convection',
                       help='experience data save dir')
    parse.add_argument('--SaveVelocityName', default='Convection1/', help='experience data set name')

    parse.add_argument('--num_person', default=26, type=int, help='number of person')

    args = parse.parse_args()

    data = tfrecords_maker(args.DirectionDatasetDir, args.DirectionDatasetName)
    data.convert_to_tfrecord(os.path.join(args.SaveDirecitonDir, args.SaveDirecitonName), args.num_person, args.SenceName)

    datav = tfrecords_maker(args.VelocityDatasetDir, args.VelocityDatasetName)
    datav.convert_to_tfrecord(os.path.join(args.SaveVelocityDir, args.SaveVelocityName), args.num_person, args.SenceName)