import tensorflow as tf
import numpy as np
import os
import datetime, time
import m4_DataReader as Reader



class Trajectors_Prediction:
    def __init__(self, sess, cfg):
        self.sess = sess
        self.cfg = cfg

        self.input_data = tf.placeholder(dtype=tf.float32, shape=[self.cfg.batch_size * self.cfg.num_gpus, 53],
                                     name='real_image')

        tensorflow_file_reader = Reader.Reader(self.cfg.dataset_dir, self.cfg.dataset_name)
        self.one_element = tensorflow_file_reader.build_dataset(self.cfg.batch_size, self.cfg.epoch, shuffle_num=10000, is_train=True)




    def train(self):
        try:
            self.saver = tf.train.Saver()
        except:
            print('one model save error....')
        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()

        self.writer = tf.summary.FileWriter('{}/{}'.format(self.cfg.log_dir,
                                                           time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))),
                                                           self.sess.graph)
        merged = tf.summary.merge_all()

        batch_data = self.sess.run(self.one_element)
        print(batch_data)
        print(batch_data.shape)



    def test(self):
        print('test....')




    def save(self, checkpoint_dir, step, model_file_name):
        model_name = "GAN.model"
        checkpoint_dir = os.path.join(checkpoint_dir, model_file_name)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir, model_folder_name):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, model_folder_name)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def load_expr_shape_pose_param(self):
        # Add ops to save and restore all the variables.
        saver_pose = tf.train.Saver(
            var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Spatial_Transformer'))
        saver_ini_shape_net = tf.train.Saver(
            var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='shapeCNN'))
        saver_ini_expr_net = tf.train.Saver(
            var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='exprCNN'))

        # Load face pose net model from Chang et al.'ICCVW17
        try:
            load_path = self.cfg.fpn_new_model_ckpt_file_path
            saver_pose.restore(self.sess, load_path)
            print('Load ' + self.cfg.fpn_new_model_ckpt_file_path + ' successful....')
        except:
            raise Exception('Load ' + self.cfg.fpn_new_model_ckpt_file_path + ' failed....')

        # load 3dmm shape and texture model from Tran et al.' CVPR2017
        try:
            load_path = self.cfg.Shape_Model_file_path
            saver_ini_shape_net.restore(self.sess, load_path)
            print('Load ' + self.cfg.Shape_Model_file_path + ' successful....')
        except:
            raise Exception('Load ' + self.cfg.Shape_Model_file_path + ' failed....')

        # load our expression net model
        try:
            load_path = self.cfg.Expression_Model_file_path
            saver_ini_expr_net.restore(self.sess, load_path)
            print('Load ' + self.cfg.Expression_Model_file_path + ' successful....')
        except:
            raise Exception('Load ' + self.cfg.Expression_Model_file_path + ' failed....')
