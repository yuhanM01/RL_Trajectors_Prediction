import numpy as np
import tensorflow as tf
import os

class Reader:
    def __init__(self, tfrecords_dir, tfrecords_name):
        '''
        :param tfrecords_dir: 存储tfrecords文件的文件夹目录
        :param tfrecords_name: 存储tfrecords文件的文件夹名称
        '''


        self.tfrecords_dir = tfrecords_dir    # model_data
        self.tfrecords_name = tfrecords_name
        file_pattern = os.path.join(self.tfrecords_dir, self.tfrecords_name) + "/*" + '.tfrecords'
        self.TfrecordFile = tf.gfile.Glob(file_pattern)

    def parser(self, serialized_example):
        """
        Introduction
        ------------
            解析tfRecord数据
        Parameters
        ----------
            serialized_example: 序列化的每条数据
        """
        features = tf.parse_single_example(
            serialized_example,
            features= {
             'person_data': tf.FixedLenFeature([], dtype = tf.string),
             }
        )
        person_data = tf.decode_raw(features['person_data'], tf.float32)
        person_data = tf.reshape(person_data, [-1, 53])

        return person_data




    def build_dataset(self, batch_size, epoch, shuffle_num=10000, is_train=True):
        """
        Introduction
        ------------
            建立数据集dataset
        Parameters
        ----------
            batch_size: batch大小
        Return
        ------
            dataset: 返回tensorflow的dataset
        """

        dataset = tf.data.TFRecordDataset(filenames = self.TfrecordFile)
        dataset = dataset.map(self.parser, num_parallel_calls = 10)
        if is_train:
            dataset = dataset.shuffle(shuffle_num).batch(batch_size).repeat(epoch)
        else:
            dataset = dataset.batch(batch_size).repeat(epoch)
        # dataset = dataset.repeat().batch(batch_size).prefetch(batch_size)
        iterator = dataset.make_one_shot_iterator()
        one_element = iterator.get_next()
        return one_element
