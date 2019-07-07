from __future__ import division, print_function, absolute_import
import os
import argparse
import tensorflow as tf
import param
from model import Trajectors_Prediction
import time

os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # 指定第  块GPU可用

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}
# TF_CPP_MIN_LOG_LEVEL 取值 0 ： 0也是默认值，输出所有信息
# TF_CPP_MIN_LOG_LEVEL 取值 1 ： 屏蔽通知信息
# TF_CPP_MIN_LOG_LEVEL 取值 2 ： 屏蔽通知信息和警告信息
# TF_CPP_MIN_LOG_LEVEL 取值 3 ： 屏蔽通知信息、警告信息和报错信息


parser = argparse.ArgumentParser()

parser.add_argument("--is_train", default=param.is_train, type=int, help="Train")
parser.add_argument("--dataset_dir", default='/home/yang/study/datasetandparam/Predestrain_dataset/comp/counterflow', type=str, help="Train data set dir")
parser.add_argument("--dataset_name", default='1_1_DirectionTfrecord', type=str, help="Train data set name")
parser.add_argument("--log_dir", default='/home/yang/study/RL_result', type=str, help="Train data label name")
parser.add_argument("--checkpoint_dir", default='/home/yang/study/RL_result', type=str, help="model save dir")
parser.add_argument("--num_gpus", default=1, type=int, help="num of gpu")
parser.add_argument("--epoch", default=20, type=int, help="epoch")
parser.add_argument("--batch_size", default=10, type=int, help="batch size for one gpus")
parser.add_argument("--lr", default=0.0001, type=float, help="learning rate of G")
parser.add_argument("--savemodel_period", default=2, type=int, help="savemodel_period")
parser.add_argument("--add_summary_period", default=20, type=int, help="add_summary_period")

cfg = parser.parse_args()

if __name__ == '__main__':

    config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = 0.9
    with tf.Session(config=config) as sess:
        if not os.path.exists(cfg.log_dir):
            os.makedirs(cfg.log_dir)


        predict = Trajectors_Prediction(sess, cfg)
        if cfg.is_train:
            predict.train()
        else:
            print('test....')

            predict.test()
