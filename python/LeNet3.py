import tensorflow as tf
from helper import conv

def LeNet3(input_tensor, keep_prob, training):
    
    conv1 = conv(input_tensor, training, filters=1, padding='VALID', name='conv1', G=1) # 28x28x1
    conv2 = conv(conv1, training, filters=20, strides=2, padding='VALID', name='conv2', G=1) # 12x12x20
    conv2 = tf.nn.dropout(conv2, keep_prob)
    conv3 = conv(conv2, training, filters=20, strides=2, padding='VALID', name='conv3', G=1) # 4x4x20
    conv4 = conv(conv3, training, filters=80, kernel_size=2, strides=2, padding='VALID', name='conv4', G=1) # 2x2x40
    conv5 = conv(conv4, training, filters=43, kernel_size=2, padding='VALID', name='conv5', G=1) # 1x1x43

    return tf.contrib.layers.flatten(conv5)