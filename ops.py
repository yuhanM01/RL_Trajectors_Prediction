import tensorflow as tf
import numpy as np

def m4_leak_relu(x, leak=0.2):
    return tf.maximum(x, leak * x, name='leak_relu')

def m4_batch_norm(input_, is_trainable):
    # try:
    output = tf.contrib.layers.batch_norm(input_, decay=0.9,
                                          updates_collections=None,
                                          epsilon=1e-5,
                                          scale=True,
                                          is_training=is_trainable)
    # except:
    #     mean, variance = tf.nn.moments(input_, axes=[0, 1, 2])
    #     _, _, _, nc = input_.get_shape().as_list()
    #     beta = tf.get_variable('beta', [nc], tf.float32,
    #                            initializer=tf.constant_initializer(0.0, tf.float32))  # [out_channels]
    #     gamma = tf.get_variable('gamma', [nc], tf.float32,
    #                             initializer=tf.constant_initializer(1.0, tf.float32))
    #     output = tf.nn.batch_normalization(input_, mean, variance, beta, gamma, 1e-5)
    return output

def m4_norm_func(input_, is_trainable, name):
    if name==None:
        output_ = input_
    elif name=='batch_norm':
        output_ = m4_batch_norm(input_, is_trainable)
    return output_

def m4_active_function(input_, active_function='relu'):
    if active_function==None:
        active = input_
    elif active_function == 'relu':
        active = tf.nn.relu(input_)
    elif active_function == 'leak_relu':
        active = m4_leak_relu(input_)
    return active






def m4_linear(input_, output, active_function=None, norm=None, get_vars_name=False, is_trainable=True,
              stddev=0.02, name='fc'):
    with tf.variable_scope(name) as scope:
        input_shape = input_.get_shape().as_list()
        w = tf.get_variable('w', [input_shape[-1], output], initializer=tf.random_normal_initializer(stddev=stddev))
        biases = tf.get_variable('biases', [output], initializer=tf.constant_initializer(0.0))
        conn = tf.matmul(input_, w) + biases
        output_ = m4_norm_func(conn, is_trainable, name=norm)
        output_ = m4_active_function(output_, active_function=active_function)
        if get_vars_name:
            vars = tf.contrib.framework.get_variables(scope)
            return output_, vars
        else:
            return output_


