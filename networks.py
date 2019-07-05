import tensorflow as tf
import numpy as np
import os
from ops import *
from utils import *
import time


class m4_RL_Trajectory_prediction:
    def __init__(self, sess, cfg):
        self.sess = sess
        self.cfg = cfg
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

    def build_Direction_model(self, x):
        a = 1

    def build_Velocity_model(self, x):
        a = 1

    def m4_DirectionNetwork(self, x):
        a = 2

    def m4_VelocityNetwork(self, x):
        a = 2