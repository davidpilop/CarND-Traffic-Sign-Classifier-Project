from os.path import join, abspath, dirname
import tensorflow as tf

from helper import conv

def AllCNN32(input_tensor, keep_prob, training):
    
    conv1 = conv(input_tensor, training, filters=96, strides=2, name='conv1')
    conv2 = conv(conv1, training, filters=96, strides=2, name='conv2')
    conv3 = conv(conv2, training, filters=96, strides=2, name='conv3')

    drop1 = tf.nn.dropout(conv3, keep_prob)

    conv4 = conv(drop1, training, filters=192, strides=2, name='conv4')
    conv5 = conv(conv4, training, filters=192, strides=2, name='conv5')
    conv6 = conv(conv5, training, filters=192, kernel_size=1, strides=2, name='conv6')

    drop2 = tf.nn.dropout(conv6, keep_prob)

    conv7 = conv(drop2, training, filters=192, kernel_size=1, name='conv7')
    conv8 = conv(conv7, training, filters=192, kernel_size=1, name='conv8')

    fcl1 = tf.contrib.layers.flatten(conv8)
    logits = tf.layers.dense(fcl1, 43)

    return logits